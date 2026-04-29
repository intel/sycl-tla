#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import argparse
import copy
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from intel_gemm_profiler.catalog import SEED_KERNELS, build_kernel_catalog
    from intel_gemm_profiler.candidates import (
        build_candidate_build_manifest,
        build_compiler_profile_probe_entries,
        build_dpas_probe_entry,
        build_phase_a_probe_entries,
        build_screening_entries,
        default_shapes,
        dry_run_shapes,
        generate_candidate_space,
        generate_confirmation_entries,
    )
    from intel_gemm_profiler.constraints import (
        apply_probe_results_to_profiles,
        apply_run_probe_constraints,
        apply_static_probe_constraints,
        default_compiler_profiles,
        default_constraints,
    )
    from intel_gemm_profiler.hw_specs import detect_probe_anomalies, resolve_hw_reference_spec
    from intel_gemm_profiler.runner import collect_environment_metadata, run_entries_with_benchmark, run_entries_with_streamk_example
    from intel_gemm_profiler.selector import build_dispatch_table, build_phase_a_summary, build_phase_b_summary, build_run_summary, write_results_csv
    from intel_gemm_profiler.utils import ensure_dir, read_json, shell_init_with_env, shell_join, write_json
    from intel_gemm_profiler.schemas import SEARCH_RUNTIME_SCHEMA
else:
    from .catalog import SEED_KERNELS, build_kernel_catalog
    from .candidates import (
        build_candidate_build_manifest,
        build_compiler_profile_probe_entries,
        build_dpas_probe_entry,
        build_phase_a_probe_entries,
        build_screening_entries,
        default_shapes,
        dry_run_shapes,
        generate_candidate_space,
        generate_confirmation_entries,
    )
    from .constraints import (
        apply_probe_results_to_profiles,
        apply_run_probe_constraints,
        apply_static_probe_constraints,
        default_compiler_profiles,
        default_constraints,
    )
    from .hw_specs import detect_probe_anomalies, resolve_hw_reference_spec
    from .runner import collect_environment_metadata, run_entries_with_benchmark, run_entries_with_streamk_example
    from .selector import build_dispatch_table, build_phase_a_summary, build_phase_b_summary, build_run_summary, write_results_csv
    from .utils import ensure_dir, read_json, shell_init_with_env, shell_join, write_json
    from .schemas import SEARCH_RUNTIME_SCHEMA


def build_compiler_flags_probe_summary(rows):
    by_profile = {}
    for row in rows:
        by_profile[row["compiler_profile_id"]] = {
            "compiler_profile_id": row["compiler_profile_id"],
            "status": row["status"],
            "avg_tflops": row["avg_tflops"],
            "avg_runtime_ms": row["avg_runtime_ms"],
            "candidate_id": row["candidate_id"],
            "shape_id": row["shape_id"],
            "log": row["stdout_log"],
        }
    grouped = {}
    for item in by_profile.values():
        grouped.setdefault(item["compiler_profile_id"].split(".")[1], []).append(item)
    selected = {}
    for candidate_class, items in grouped.items():
        passed = [item for item in items if item["status"] == "pass"]
        if passed:
            selected[candidate_class] = max(passed, key=lambda item: float(item["avg_tflops"] or 0.0))["compiler_profile_id"]
    return {"results": list(by_profile.values()), "selected_profile_ids": selected}


def run_phase_a_probe(args, shapes_doc, base_constraints, profiles, reports_dir, configs_dir, manifests_dir, logs_dir):
    env_caps = collect_environment_metadata(args.shell_init, args.benchmark_exe, args.streamk_example_exe, cwd=args.cwd)
    static_constraints = apply_static_probe_constraints(base_constraints, env_caps)
    hw_spec = resolve_hw_reference_spec(static_constraints["device_arch"], getattr(args, "hw_spec_id", ""))
    static_candidate_space = generate_candidate_space(shapes_doc, static_constraints, profiles, allowed_runners=("benchmark", "streamk_example"))
    probe_rows = []
    probe_logs = []
    probe_commands = []
    probe_entries = build_phase_a_probe_entries(shapes_doc, static_candidate_space)
    effective_probe_mode = args.probe_mode
    if effective_probe_mode == "auto":
        effective_probe_mode = "static" if args.skip_run else "run"
    if effective_probe_mode == "run" and not args.skip_run and probe_entries:
        probe_benchmark_entries = [entry for entry in probe_entries if entry["candidate"].get("runner", "benchmark") == "benchmark"]
        probe_streamk_entries = [entry for entry in probe_entries if entry["candidate"].get("runner") == "streamk_example"]
        if probe_benchmark_entries:
            probe_log = logs_dir / "probe.log"
            rows, command = run_entries_with_benchmark(probe_benchmark_entries, configs_dir / "probe.in", manifests_dir / "probe_manifest.json", probe_log, args.benchmark_exe, cwd=args.cwd, shell_init=args.shell_init, timeout=args.timeout)
            probe_rows.extend(rows)
            probe_logs.append(str(probe_log))
            probe_commands.append(shell_join(command))
        if probe_streamk_entries:
            rows, commands = run_entries_with_streamk_example(probe_streamk_entries, logs_dir, args.streamk_example_exe, cwd=args.cwd, shell_init=args.shell_init, timeout=args.timeout)
            probe_rows.extend(rows)
            probe_logs.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in probe_streamk_entries)
            probe_commands.extend(commands)
    dpas_probe = {"status": "skipped", "reason": "probe mode disabled or benchmark unavailable"}
    compiler_flags_probe = {"results": [], "selected_profile_ids": {}}
    if effective_probe_mode == "run" and not args.skip_run and env_caps["executables"]["benchmark_available"]:
        dpas_entry = build_dpas_probe_entry(shapes_doc, static_candidate_space)
        if dpas_entry:
            dpas_log = logs_dir / "dpas_probe.log"
            rows, command = run_entries_with_benchmark([dpas_entry], configs_dir / "dpas_probe.in", manifests_dir / "dpas_probe_manifest.json", dpas_log, args.benchmark_exe, cwd=args.cwd, shell_init=args.shell_init, timeout=args.timeout)
            if rows:
                probe_rows.extend(rows)
                probe_logs.append(str(dpas_log))
                probe_commands.append(shell_join(command))
                row = rows[0]
                dpas_probe = {"status": row["status"], "candidate_id": row["candidate_id"], "shape_id": row["shape_id"], "avg_tflops": row["avg_tflops"], "avg_runtime_ms": row["avg_runtime_ms"], "log": str(dpas_log)}
            else:
                dpas_probe = {"status": "fail", "reason": "missing benchmark row", "log": str(dpas_log)}
        compiler_probe_entries = build_compiler_profile_probe_entries(shapes_doc, static_candidate_space, profiles)
        compiler_probe_rows = []
        for entry in compiler_probe_entries:
            profile = next(profile for profile in profiles["profiles"] if profile["compiler_profile_id"] == entry["compiler_profile_probe_id"])
            compiler_log = logs_dir / f"{entry['compiler_profile_probe_id'].replace('.', '_')}.log"
            rows, command = run_entries_with_benchmark([entry], configs_dir / f"{entry['compiler_profile_probe_id'].replace('.', '_')}.in", manifests_dir / f"{entry['compiler_profile_probe_id'].replace('.', '_')}_manifest.json", compiler_log, args.benchmark_exe, cwd=args.cwd, shell_init=shell_init_with_env(args.shell_init, profile.get("env", {})), timeout=args.timeout)
            compiler_probe_rows.extend(rows)
            probe_logs.append(str(compiler_log))
            probe_commands.append(shell_join(command))
        compiler_flags_probe = build_compiler_flags_probe_summary(compiler_probe_rows)
    anomaly_report = detect_probe_anomalies(probe_rows, shapes_doc, static_candidate_space, hw_spec) if probe_rows else {"hw_spec": hw_spec["device_id"], "peak_tflops": hw_spec.get("peak_xmx_tflops", 0.0), "anomalies": [], "auto_block_rules": []}
    constraints = apply_run_probe_constraints(static_constraints, probe_rows, anomaly_report=anomaly_report) if probe_rows else static_constraints
    env_caps["probe_mode"] = effective_probe_mode
    env_caps["hw_reference_spec_id"] = hw_spec["device_id"]
    env_caps["hw_reference_spec"] = hw_spec
    env_caps["constraint_source"] = constraints["constraint_source"]
    env_caps["dpas_baseline_probe"] = dpas_probe
    env_caps["compiler_flags_probe"] = compiler_flags_probe
    env_caps["anomaly_report"] = anomaly_report
    env_caps["probe_results"] = [{"candidate_id": row["candidate_id"], "shape_id": row["shape_id"], "status": row["status"], "avg_tflops": row["avg_tflops"], "split_k": row["split_k"]} for row in probe_rows]
    verified_hw_caps_path = reports_dir / "verified_hw_caps.json"
    write_json(verified_hw_caps_path, env_caps)
    return constraints, env_caps, verified_hw_caps_path, probe_rows, probe_logs, probe_commands


def workflow(args):
    workspace = ensure_dir(Path(args.workspace).resolve())
    inputs_dir = ensure_dir(workspace / "inputs")
    generated_dir = ensure_dir(workspace / "generated")
    configs_dir = ensure_dir(generated_dir / "configs")
    manifests_dir = ensure_dir(generated_dir / "manifests")
    logs_dir = ensure_dir(workspace / "logs")
    reports_dir = ensure_dir(workspace / "reports")
    profiles = read_json(args.compiler_profiles_json) if args.compiler_profiles_json else default_compiler_profiles()
    dry_run_mode = getattr(args, "dry_run", False)
    shapes_doc = read_json(args.shapes_json) if args.shapes_json else (dry_run_shapes(args.dtype) if dry_run_mode else default_shapes(args.dtype))
    base_constraints = read_json(args.constraints_json) if args.constraints_json else default_constraints()
    top_k = min(args.top_k, 1) if dry_run_mode else args.top_k
    confirm_runs = 0 if dry_run_mode else args.confirm_runs
    probe_mode = "off" if dry_run_mode else args.probe_mode
    probe_rows = []
    probe_logs = []
    probe_commands = []
    benchmark_commands = []
    if args.constraints_json or probe_mode == "off":
        constraints = copy.deepcopy(base_constraints)
        env_caps = collect_environment_metadata(args.shell_init, args.benchmark_exe, args.streamk_example_exe, cwd=args.cwd)
        hw_spec = resolve_hw_reference_spec(constraints["device_arch"], getattr(args, "hw_spec_id", ""))
        env_caps["probe_mode"] = "dry_run_off" if dry_run_mode else ("off" if probe_mode == "off" else "external_constraints")
        env_caps["hw_reference_spec_id"] = hw_spec["device_id"]
        env_caps["hw_reference_spec"] = hw_spec
        env_caps["constraint_source"] = constraints["constraint_source"]
        env_caps["anomaly_report"] = {"hw_spec": hw_spec["device_id"], "peak_tflops": hw_spec.get("peak_xmx_tflops", 0.0), "anomalies": [], "auto_block_rules": []}
        env_caps["probe_results"] = []
        verified_hw_caps_path = reports_dir / "verified_hw_caps.json"
        write_json(verified_hw_caps_path, env_caps)
    else:
        constraints, env_caps, verified_hw_caps_path, probe_rows, probe_logs, probe_commands = run_phase_a_probe(args, shapes_doc, base_constraints, profiles, reports_dir, configs_dir, manifests_dir, logs_dir)
        profiles = apply_probe_results_to_profiles(profiles, env_caps.get("compiler_flags_probe", {}))
    write_json(inputs_dir / "safe_search_constraints.json", constraints)
    write_json(inputs_dir / "compiler_profiles.json", profiles)
    write_json(inputs_dir / "gemm_target_shapes.json", shapes_doc)
    write_json(inputs_dir / "search_runtime_schema.json", SEARCH_RUNTIME_SCHEMA)
    kernel_catalog = build_kernel_catalog(dtypes=sorted({shape["dtype_a"] for shape in shapes_doc["shapes"]}), allowed_runners=("benchmark", "streamk_example"))
    write_json(reports_dir / "kernel_catalog.json", kernel_catalog)
    candidate_space = generate_candidate_space(shapes_doc, constraints, profiles, allowed_runners=("benchmark",))
    write_json(reports_dir / "gemm_candidate_space.json", candidate_space)
    write_json(reports_dir / "bmg_safe_candidates.json", candidate_space)
    write_json(reports_dir / "candidate_build_manifest.json", build_candidate_build_manifest(candidate_space))
    screening_entries = build_screening_entries(shapes_doc, candidate_space)
    all_rows = list(probe_rows)
    log_paths = list(probe_logs)
    benchmark_commands.extend(probe_commands)
    if not args.skip_run:
        screening_benchmark_entries = [entry for entry in screening_entries if entry["candidate"].get("runner", "benchmark") == "benchmark"]
        screening_streamk_entries = [entry for entry in screening_entries if entry["candidate"].get("runner") == "streamk_example"]
        screening_rows = []
        if screening_benchmark_entries:
            screening_log = logs_dir / "screening.log"
            rows, command = run_entries_with_benchmark(screening_benchmark_entries, configs_dir / "screening.in", manifests_dir / "screening_manifest.json", screening_log, args.benchmark_exe, cwd=args.cwd, shell_init=args.shell_init, timeout=args.timeout)
            screening_rows.extend(rows)
            log_paths.append(str(screening_log))
            benchmark_commands.append(shell_join(command))
        if screening_streamk_entries:
            rows, commands = run_entries_with_streamk_example(screening_streamk_entries, logs_dir, args.streamk_example_exe, cwd=args.cwd, shell_init=args.shell_init, timeout=args.timeout)
            screening_rows.extend(rows)
            log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in screening_streamk_entries)
            benchmark_commands.extend(commands)
        all_rows.extend(screening_rows)
        if confirm_runs > 0:
            confirm_entries = generate_confirmation_entries(screening_rows, candidate_space, shapes_doc, top_k=top_k, confirm_runs=confirm_runs)
            if confirm_entries:
                confirm_benchmark_entries = [entry for entry in confirm_entries if entry["candidate"].get("runner", "benchmark") == "benchmark"]
                confirm_streamk_entries = [entry for entry in confirm_entries if entry["candidate"].get("runner") == "streamk_example"]
                confirm_rows = []
                if confirm_benchmark_entries:
                    confirm_log = logs_dir / "confirm.log"
                    rows, command = run_entries_with_benchmark(confirm_benchmark_entries, configs_dir / "confirm.in", manifests_dir / "confirm_manifest.json", confirm_log, args.benchmark_exe, cwd=args.cwd, shell_init=args.shell_init, timeout=args.timeout)
                    confirm_rows.extend(rows)
                    log_paths.append(str(confirm_log))
                    benchmark_commands.append(shell_join(command))
                if confirm_streamk_entries:
                    rows, commands = run_entries_with_streamk_example(confirm_streamk_entries, logs_dir, args.streamk_example_exe, cwd=args.cwd, shell_init=args.shell_init, timeout=args.timeout)
                    confirm_rows.extend(rows)
                    log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in confirm_streamk_entries)
                    benchmark_commands.extend(commands)
                all_rows.extend(confirm_rows)
    write_results_csv(all_rows, reports_dir / "gemm_profile_results.csv")
    dispatch_table = build_dispatch_table(all_rows, shapes_doc, top_k=top_k, confirm_runs=confirm_runs, close_call_threshold=args.close_call_threshold)
    write_json(reports_dir / "gemm_dispatch_table.json", dispatch_table)
    write_json(reports_dir / "optimal_dispatch_table.json", dispatch_table)
    summary = build_run_summary(all_rows, dispatch_table, benchmark_commands, log_paths)
    write_json(reports_dir / "run_summary.json", summary)
    write_json(reports_dir / "phase_a_summary.json", build_phase_a_summary(env_caps, constraints, probe_rows))
    write_json(reports_dir / "phase_b_summary.json", build_phase_b_summary(candidate_space, dispatch_table, summary))
    return {
        "workspace": str(workspace),
        "search_runtime_schema": str(inputs_dir / "search_runtime_schema.json"),
        "kernel_catalog": str(reports_dir / "kernel_catalog.json"),
        "candidate_space": str(reports_dir / "gemm_candidate_space.json"),
        "build_manifest": str(reports_dir / "candidate_build_manifest.json"),
        "safe_candidates": str(reports_dir / "bmg_safe_candidates.json"),
        "verified_hw_caps": str(verified_hw_caps_path),
        "results_csv": str(reports_dir / "gemm_profile_results.csv"),
        "dispatch_table": str(reports_dir / "gemm_dispatch_table.json"),
        "optimal_dispatch_table": str(reports_dir / "optimal_dispatch_table.json"),
        "phase_a_summary": str(reports_dir / "phase_a_summary.json"),
        "phase_b_summary": str(reports_dir / "phase_b_summary.json"),
        "summary": str(reports_dir / "run_summary.json"),
        "dry_run": dry_run_mode,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Intel GEMM profiler MVP runner for non-legacy registered RCR kernels.")
    parser.add_argument("--workspace", required=True, help="Workspace directory for generated files and reports.")
    parser.add_argument("--benchmark-exe", default="./build/benchmarks/gemm/cutlass_benchmarks_gemm_sycl", help="Benchmark executable to run.")
    parser.add_argument("--streamk-example-exe", default="./build/examples/03_bmg_gemm_streamk/03_bmg_gemm_streamk", help="StreamK example executable used for split-k candidates.")
    parser.add_argument("--cwd", default=None, help="Working directory for the benchmark subprocess.")
    parser.add_argument("--shell-init", default="", help="Optional shell snippet executed before the benchmark command, e.g. 'source /home/intel/.bashrc && source /opt/intel/oneapi/setvars.sh'.")
    parser.add_argument("--dtype", choices=sorted(SEED_KERNELS.keys()), default="bf16", help="Default dtype preset.")
    parser.add_argument("--probe-mode", choices=["auto", "off", "static", "run"], default="auto", help="Phase A constraint probe mode. 'auto' runs representative probes unless --skip-run is set.")
    parser.add_argument("--shapes-json", default="", help="Optional path to gemm_target_shapes.json.")
    parser.add_argument("--constraints-json", default="", help="Optional path to safe_search_constraints.json.")
    parser.add_argument("--compiler-profiles-json", default="", help="Optional path to compiler_profiles.json.")
    parser.add_argument("--hw-spec-id", default="", help="Optional hardware reference spec id override, e.g. 'bmg_g21'.")
    parser.add_argument("--skip-run", action="store_true", help="Only emit generated artifacts without invoking the benchmark.")
    parser.add_argument("--dry-run", action="store_true", help="Run a minimal benchmark-backed screening smoke with a tiny shape set and no confirmation.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-subprocess timeout in seconds for benchmark and example runs.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k candidates kept for confirmation.")
    parser.add_argument("--confirm-runs", type=int, default=3, help="Number of confirmation attempts for top-k candidates.")
    parser.add_argument("--close-call-threshold", type=float, default=3.0, help="Gap threshold in percent for close-call labeling.")
    return parser


def main():
    args = build_parser().parse_args()
    print(json.dumps(workflow(args), indent=2))


if __name__ == "__main__":
    main()
