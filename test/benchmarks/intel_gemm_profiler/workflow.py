#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import argparse
import copy
import json
import statistics
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
        selected_runtime_env,
    )
    from intel_gemm_profiler.hw_specs import detect_probe_anomalies, resolve_hw_reference_spec
    from intel_gemm_profiler.ali_dataset import build_ali_gemm_docs
    from intel_gemm_profiler.runner import collect_environment_metadata, run_benchmark, run_entries_with_benchmark, run_entries_with_streamk_example
    from intel_gemm_profiler.selector import build_dispatch_table, build_phase_a_summary, build_phase_b_summary, build_reference_comparison, build_run_summary, write_results_csv
    from intel_gemm_profiler.utils import ensure_dir, read_json, resolve_executable, shell_init_with_env, shell_join, write_json
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
        selected_runtime_env,
    )
    from .hw_specs import detect_probe_anomalies, resolve_hw_reference_spec
    from .ali_dataset import build_ali_gemm_docs
    from .runner import collect_environment_metadata, run_benchmark, run_entries_with_benchmark, run_entries_with_streamk_example
    from .selector import build_dispatch_table, build_phase_a_summary, build_phase_b_summary, build_reference_comparison, build_run_summary, write_results_csv
    from .utils import ensure_dir, read_json, resolve_executable, shell_init_with_env, shell_join, write_json
    from .schemas import SEARCH_RUNTIME_SCHEMA


def build_compiler_flags_probe_summary(rows, profiles=None):
    profile_class_map = {
        profile["compiler_profile_id"]: profile.get("candidate_class", "")
        for profile in (profiles or {}).get("profiles", [])
    }
    by_profile = {}
    for row in rows:
        profile_id = row["compiler_profile_id"]
        by_profile.setdefault(profile_id, []).append(
            {
                "compiler_profile_id": profile_id,
                "candidate_class": row.get("candidate_class", profile_class_map.get(profile_id, "")),
                "status": row["status"],
                "avg_tflops": row["avg_tflops"],
                "avg_runtime_ms": row["avg_runtime_ms"],
                "candidate_id": row["candidate_id"],
                "shape_id": row["shape_id"],
                "log": row["stdout_log"],
            }
        )
    summarized = []
    for profile_id, items in by_profile.items():
        passed = [item for item in items if item["status"] == "pass"]
        best_pass = max(passed, key=lambda item: float(item["avg_tflops"] or 0.0), default=None)
        status = "pass" if passed else items[0]["status"]
        avg_tflops = ""
        avg_runtime_ms = ""
        if passed:
            tflops_values = [float(item["avg_tflops"]) for item in passed if item["avg_tflops"] != ""]
            runtime_values = [float(item["avg_runtime_ms"]) for item in passed if item["avg_runtime_ms"] != ""]
            avg_tflops = str(statistics.median(tflops_values)) if tflops_values else ""
            avg_runtime_ms = str(statistics.median(runtime_values)) if runtime_values else ""
        reference = best_pass or items[0]
        summarized.append(
            {
                "compiler_profile_id": profile_id,
                "candidate_class": reference["candidate_class"],
                "status": status,
                "avg_tflops": avg_tflops,
                "avg_runtime_ms": avg_runtime_ms,
                "candidate_id": reference["candidate_id"],
                "shape_id": reference["shape_id"],
                "log": reference["log"],
                "samples": len(items),
            }
        )
    grouped = {}
    for item in summarized:
        grouped.setdefault(item["candidate_class"], []).append(item)
    selected = {}
    for candidate_class, items in grouped.items():
        passed = [item for item in items if item["status"] == "pass"]
        if passed:
            selected[candidate_class] = max(passed, key=lambda item: float(item["avg_tflops"] or 0.0))["compiler_profile_id"]
    return {"results": summarized, "selected_profile_ids": selected}


def empty_anomaly_report(hw_spec):
    return {
        "hw_spec": hw_spec["device_id"],
        "hw_spec_calibration_status": hw_spec.get("calibration_status", "unknown"),
        "peak_tflops": hw_spec.get("peak_xmx_tflops", 0.0),
        "anomalies": [],
        "auto_block_rules": [],
    }


def load_compiled_kernel_list(path):
    if not path:
        return None
    kernels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if item.startswith("^") and item.endswith("$"):
            item = item[1:-1]
        kernels.append(item)
    return kernels


def filter_candidate_space_by_compiled_kernels(candidate_space, compiled_kernels):
    if compiled_kernels is None:
        return candidate_space
    compiled = set(compiled_kernels)
    filtered = copy.deepcopy(candidate_space)
    filtered["candidates"] = [
        candidate for candidate in candidate_space["candidates"]
        if candidate.get("runner", "benchmark") != "benchmark" or candidate["kernel_id"] in compiled
    ]
    filtered["compiled_kernel_filter"] = {
        "source": "compiled_kernel_list",
        "kernel_count": len(compiled),
        "matched_candidate_count": len(filtered["candidates"]),
    }
    if candidate_space["candidates"] and not filtered["candidates"]:
        raise ValueError("Compiled kernel list does not match any generated benchmark candidates.")
    return filtered


def benchmark_exe_for_build_plan(build_dir, build_target):
    return str(Path(build_dir) / "benchmarks" / "gemm" / build_target)


def build_candidate_build_plan(
    build_manifest,
    source_dir,
    build_dir,
    kernel_filter_path,
    googlebenchmark_dir=None,
    cmake_cxx_compiler="",
):
    cmake_config = build_manifest["cmake_config"]
    cmake_vars = dict(cmake_config["cmake_vars"])
    kernel_filter_var = cmake_config["kernel_filter_cmake_var"]
    cmake_vars[kernel_filter_var] = str(kernel_filter_path)
    if googlebenchmark_dir:
        cmake_vars["GOOGLEBENCHMARK_DIR"] = str(googlebenchmark_dir)
    if cmake_cxx_compiler:
        cmake_vars["CMAKE_CXX_COMPILER"] = cmake_cxx_compiler
    configure_command = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    configure_command.extend(f"-D{name}={value}" for name, value in sorted(cmake_vars.items()))
    build_command = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        cmake_config["build_target"],
        "--parallel",
    ]
    return {
        "schema_version": build_manifest["schema_version"],
        "generated_at": build_manifest["generated_at"],
        "build_target": cmake_config["build_target"],
        "source_dir": str(source_dir),
        "build_dir": str(build_dir),
        "benchmark_exe": benchmark_exe_for_build_plan(build_dir, cmake_config["build_target"]),
        "kernel_filter_file": str(kernel_filter_path),
        "googlebenchmark_dir": str(googlebenchmark_dir) if googlebenchmark_dir else "",
        "cmake_cxx_compiler": cmake_cxx_compiler,
        "selected_kernel_count": build_manifest["selected_kernel_count"],
        "cmake_vars": cmake_vars,
        "configure_command": configure_command,
        "build_command": build_command,
        "configure_command_line": shell_join(configure_command),
        "build_command_line": shell_join(build_command),
    }


def execute_candidate_build_plan(build_plan, log_dir, shell_init="", timeout=None):
    ensure_dir(Path(log_dir))
    steps = [
        ("configure", build_plan["configure_command"], Path(log_dir) / "candidate_build_configure.log"),
        ("build", build_plan["build_command"], Path(log_dir) / "candidate_build.log"),
    ]
    results = []
    for step, command, log_path in steps:
        process, timed_out, timeout_reason = run_benchmark(command, log_path, shell_init=shell_init, timeout=timeout)
        status = "timeout" if timed_out else ("pass" if process.returncode == 0 else "fail")
        item = {
            "step": step,
            "status": status,
            "returncode": process.returncode,
            "command": shell_join(command),
            "log": str(log_path),
        }
        if timed_out:
            item["timeout_reason"] = timeout_reason
        results.append(item)
        if status != "pass":
            raise RuntimeError(f"Candidate benchmark {step} failed with status {status}. See {log_path}.")
    return {
        "schema_version": build_plan["schema_version"],
        "generated_at": build_plan["generated_at"],
        "status": "pass",
        "build_target": build_plan["build_target"],
        "benchmark_exe": build_plan["benchmark_exe"],
        "steps": results,
    }


def validate_candidate_auto_build_mode(args, dry_run_mode, probe_mode):
    if not args.build_candidate_benchmark or dry_run_mode or args.skip_run or args.constraints_json:
        return
    if probe_mode not in {"auto", "run"}:
        return
    if resolve_executable(args.benchmark_exe, cwd=args.cwd):
        return
    raise ValueError(
        "--build-candidate-benchmark builds the generated benchmark after Phase A. "
        "Use --probe-mode=off or --constraints-json when no prebuilt --benchmark-exe is available for Phase A probes."
    )


def load_target_shapes_and_reference(args, dry_run_mode):
    if args.ali_workbook:
        if args.shapes_json:
            raise ValueError("--ali-workbook and --shapes-json are mutually exclusive.")
        if args.reference_json:
            raise ValueError("--ali-workbook and --reference-json are mutually exclusive.")
        shapes_doc, reference_doc = build_ali_gemm_docs(args.ali_workbook)
        return limit_shapes_and_reference(shapes_doc, reference_doc, args.max_shapes)
    shapes_doc = read_json(args.shapes_json) if args.shapes_json else (dry_run_shapes(args.dtype) if dry_run_mode else default_shapes(args.dtype))
    reference_doc = read_json(args.reference_json) if args.reference_json else None
    return limit_shapes_and_reference(shapes_doc, reference_doc, args.max_shapes)


def limit_shapes_and_reference(shapes_doc, reference_doc=None, max_shapes=0):
    if max_shapes is None or max_shapes == 0:
        return shapes_doc, reference_doc
    if max_shapes < 0:
        raise ValueError("--max-shapes must be non-negative.")
    limited_shapes_doc = copy.deepcopy(shapes_doc)
    selected_shapes = limited_shapes_doc.get("shapes", [])[:max_shapes]
    limited_shapes_doc["shapes"] = selected_shapes
    limited_shapes_doc["shape_limit"] = max_shapes
    limited_shapes_doc["unlimited_shape_count"] = len(shapes_doc.get("shapes", []))
    if reference_doc is None:
        return limited_shapes_doc, None
    selected_shape_ids = {shape["shape_id"] for shape in selected_shapes}
    selected_shape_keys = {
        (shape.get("dtype_a"), shape.get("m"), shape.get("n"), shape.get("k"))
        for shape in selected_shapes
    }
    limited_reference_doc = copy.deepcopy(reference_doc)
    limited_reference_doc["entries"] = [
        entry for entry in limited_reference_doc.get("entries", [])
        if entry.get("shape_id") in selected_shape_ids
    ]
    limited_reference_doc["skipped_entries"] = [
        entry for entry in limited_reference_doc.get("skipped_entries", [])
        if (entry.get("dtype"), entry.get("m"), entry.get("n"), entry.get("k")) in selected_shape_keys
    ]
    limited_reference_doc["shape_limit"] = max_shapes
    limited_reference_doc["unlimited_reference_entries"] = len(reference_doc.get("entries", []))
    return limited_shapes_doc, limited_reference_doc


def run_phase_a_probe(args, shapes_doc, base_constraints, profiles, reports_dir, configs_dir, manifests_dir, logs_dir):
    base_runtime_shell_init = shell_init_with_env(args.shell_init, selected_runtime_env(profiles))
    env_caps = collect_environment_metadata(args.shell_init, args.benchmark_exe, args.streamk_example_exe, cwd=args.cwd)
    static_constraints = apply_static_probe_constraints(base_constraints, env_caps)
    hw_spec = resolve_hw_reference_spec(static_constraints["device_arch"], getattr(args, "hw_spec_id", ""))
    allowed_runners = ("benchmark", "streamk_example") if env_caps["executables"].get("streamk_example_available") else ("benchmark",)
    static_candidate_space = generate_candidate_space(shapes_doc, static_constraints, profiles, allowed_runners=allowed_runners)
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
            rows, command = run_entries_with_benchmark(probe_benchmark_entries, configs_dir / "probe.in", manifests_dir / "probe_manifest.json", probe_log, args.benchmark_exe, cwd=args.cwd, shell_init=base_runtime_shell_init, timeout=args.timeout)
            probe_rows.extend(rows)
            probe_logs.append(str(probe_log))
            probe_commands.append(shell_join(command))
        if probe_streamk_entries:
            rows, commands = run_entries_with_streamk_example(probe_streamk_entries, logs_dir, args.streamk_example_exe, cwd=args.cwd, shell_init=base_runtime_shell_init, timeout=args.timeout)
            probe_rows.extend(rows)
            probe_logs.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in probe_streamk_entries)
            probe_commands.extend(commands)
    dpas_probe = {"status": "skipped", "reason": "probe mode disabled or benchmark unavailable"}
    compiler_flags_probe = {"results": [], "selected_profile_ids": {}}
    if effective_probe_mode == "run" and not args.skip_run and env_caps["executables"]["benchmark_available"]:
        dpas_entry = build_dpas_probe_entry(shapes_doc, static_candidate_space)
        if dpas_entry:
            dpas_log = logs_dir / "dpas_probe.log"
            rows, command = run_entries_with_benchmark([dpas_entry], configs_dir / "dpas_probe.in", manifests_dir / "dpas_probe_manifest.json", dpas_log, args.benchmark_exe, cwd=args.cwd, shell_init=base_runtime_shell_init, timeout=args.timeout)
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
            runtime_shell_init = shell_init_with_env(args.shell_init, selected_runtime_env(profiles, profile))
            rows, command = run_entries_with_benchmark([entry], configs_dir / f"{entry['compiler_profile_probe_id'].replace('.', '_')}.in", manifests_dir / f"{entry['compiler_profile_probe_id'].replace('.', '_')}_manifest.json", compiler_log, args.benchmark_exe, cwd=args.cwd, shell_init=runtime_shell_init, timeout=args.timeout)
            compiler_probe_rows.extend(rows)
            probe_logs.append(str(compiler_log))
            probe_commands.append(shell_join(command))
        compiler_flags_probe = build_compiler_flags_probe_summary(compiler_probe_rows, profiles)
    anomaly_report = detect_probe_anomalies(probe_rows, shapes_doc, static_candidate_space, hw_spec) if probe_rows else empty_anomaly_report(hw_spec)
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
    shapes_doc, reference_doc = load_target_shapes_and_reference(args, dry_run_mode)
    base_constraints = read_json(args.constraints_json) if args.constraints_json else default_constraints()
    top_k = min(args.top_k, 1) if dry_run_mode else args.top_k
    confirm_runs = 0 if dry_run_mode else args.confirm_runs
    probe_mode = "off" if dry_run_mode else args.probe_mode
    validate_candidate_auto_build_mode(args, dry_run_mode, probe_mode)
    probe_rows = []
    probe_logs = []
    probe_commands = []
    benchmark_commands = []
    base_runtime_shell_init = shell_init_with_env(args.shell_init, selected_runtime_env(profiles))
    if args.constraints_json or probe_mode == "off":
        constraints = copy.deepcopy(base_constraints)
        env_caps = collect_environment_metadata(args.shell_init, args.benchmark_exe, args.streamk_example_exe, cwd=args.cwd)
        hw_spec = resolve_hw_reference_spec(constraints["device_arch"], getattr(args, "hw_spec_id", ""))
        env_caps["probe_mode"] = "dry_run_off" if dry_run_mode else ("off" if probe_mode == "off" else "external_constraints")
        env_caps["hw_reference_spec_id"] = hw_spec["device_id"]
        env_caps["hw_reference_spec"] = hw_spec
        env_caps["constraint_source"] = constraints["constraint_source"]
        env_caps["anomaly_report"] = empty_anomaly_report(hw_spec)
        env_caps["probe_results"] = []
        verified_hw_caps_path = reports_dir / "verified_hw_caps.json"
        write_json(verified_hw_caps_path, env_caps)
    else:
        constraints, env_caps, verified_hw_caps_path, probe_rows, probe_logs, probe_commands = run_phase_a_probe(args, shapes_doc, base_constraints, profiles, reports_dir, configs_dir, manifests_dir, logs_dir)
        profiles = apply_probe_results_to_profiles(profiles, env_caps.get("compiler_flags_probe", {}))
    allowed_runners = ("benchmark", "streamk_example") if env_caps["executables"].get("streamk_example_available") else ("benchmark",)
    write_json(inputs_dir / "safe_search_constraints.json", constraints)
    write_json(inputs_dir / "compiler_profiles.json", profiles)
    write_json(inputs_dir / "gemm_target_shapes.json", shapes_doc)
    reference_doc_path = reports_dir / "ali_reference.json"
    if reference_doc is not None:
        write_json(reference_doc_path, reference_doc)
    write_json(inputs_dir / "search_runtime_schema.json", SEARCH_RUNTIME_SCHEMA)
    kernel_catalog = build_kernel_catalog(
        dtypes=sorted({shape["dtype_a"] for shape in shapes_doc["shapes"]}),
        allowed_runners=allowed_runners,
        catalog_path=Path(args.kernel_catalog_path) if args.kernel_catalog_path else None,
        catalog_source=args.kernel_catalog_source,
        generator_arch=args.generator_arch,
        generator_instantiation_level=args.generator_instantiation_level,
    )
    write_json(reports_dir / "kernel_catalog.json", kernel_catalog)
    candidate_space = generate_candidate_space(
        shapes_doc,
        constraints,
        profiles,
        allowed_runners=allowed_runners,
        catalog_path=Path(args.kernel_catalog_path) if args.kernel_catalog_path else None,
        catalog_source=args.kernel_catalog_source,
        generator_arch=args.generator_arch,
        generator_instantiation_level=args.generator_instantiation_level,
    )
    candidate_space = filter_candidate_space_by_compiled_kernels(
        candidate_space,
        load_compiled_kernel_list(args.compiled_kernel_list),
    )
    write_json(reports_dir / "gemm_candidate_space.json", candidate_space)
    write_json(reports_dir / "bmg_safe_candidates.json", candidate_space)
    build_manifest = build_candidate_build_manifest(candidate_space)
    write_json(reports_dir / "candidate_build_manifest.json", build_manifest)
    selected_kernel_list_path = reports_dir / "selected_kernel_list.txt"
    selected_kernel_filter_path = reports_dir / "selected_kernel_filter.list"
    candidate_build_cmake_config_path = reports_dir / "candidate_build_cmake_config.json"
    candidate_build_plan_path = reports_dir / "candidate_build_plan.json"
    selected_kernel_list_path.write_text("\n".join(build_manifest["selected_kernel_list"]) + "\n", encoding="utf-8")
    selected_kernel_filter_path.write_text("\n".join(build_manifest["kernel_filter_file"]["lines"]) + "\n", encoding="utf-8")
    write_json(candidate_build_cmake_config_path, build_manifest["cmake_config"])
    source_dir = Path(args.cmake_source_dir).resolve() if args.cmake_source_dir else (Path(args.cwd).resolve() if args.cwd else Path.cwd().resolve())
    build_dir = Path(args.benchmark_build_dir).resolve() if args.benchmark_build_dir else workspace / "build" / "candidate_benchmarks"
    googlebenchmark_dir = Path(args.googlebenchmark_dir).resolve() if args.googlebenchmark_dir else None
    candidate_build_plan = build_candidate_build_plan(
        build_manifest,
        source_dir,
        build_dir,
        selected_kernel_filter_path,
        googlebenchmark_dir,
        args.cmake_cxx_compiler,
    )
    write_json(candidate_build_plan_path, candidate_build_plan)
    candidate_build_summary_path = reports_dir / "candidate_build_summary.json"
    candidate_build_summary = {"status": "not_run", "reason": "build_candidate_benchmark disabled"}
    effective_benchmark_exe = args.benchmark_exe
    if args.build_candidate_benchmark:
        candidate_build_summary = execute_candidate_build_plan(
            candidate_build_plan,
            logs_dir,
            shell_init=args.shell_init,
            timeout=args.timeout,
        )
        write_json(candidate_build_summary_path, candidate_build_summary)
        effective_benchmark_exe = candidate_build_plan["benchmark_exe"]
        env_caps["executables"]["benchmark_exe"] = effective_benchmark_exe
        env_caps["executables"]["benchmark_available"] = True
        env_caps["candidate_build_summary"] = candidate_build_summary
        write_json(verified_hw_caps_path, env_caps)
    else:
        write_json(candidate_build_summary_path, candidate_build_summary)
    screening_entries = build_screening_entries(shapes_doc, candidate_space)
    all_rows = list(probe_rows)
    log_paths = list(probe_logs)
    benchmark_commands.extend(probe_commands)
    if candidate_build_summary.get("status") == "pass":
        log_paths.extend(step["log"] for step in candidate_build_summary["steps"])
        benchmark_commands.extend(step["command"] for step in candidate_build_summary["steps"])
    if not args.skip_run:
        screening_benchmark_entries = [entry for entry in screening_entries if entry["candidate"].get("runner", "benchmark") == "benchmark"]
        screening_streamk_entries = [entry for entry in screening_entries if entry["candidate"].get("runner") == "streamk_example"]
        screening_rows = []
        if screening_benchmark_entries:
            screening_log = logs_dir / "screening.log"
            rows, command = run_entries_with_benchmark(screening_benchmark_entries, configs_dir / "screening.in", manifests_dir / "screening_manifest.json", screening_log, effective_benchmark_exe, cwd=args.cwd, shell_init=base_runtime_shell_init, timeout=args.timeout)
            screening_rows.extend(rows)
            log_paths.append(str(screening_log))
            benchmark_commands.append(shell_join(command))
        if screening_streamk_entries:
            rows, commands = run_entries_with_streamk_example(screening_streamk_entries, logs_dir, args.streamk_example_exe, cwd=args.cwd, shell_init=base_runtime_shell_init, timeout=args.timeout)
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
                    rows, command = run_entries_with_benchmark(confirm_benchmark_entries, configs_dir / "confirm.in", manifests_dir / "confirm_manifest.json", confirm_log, effective_benchmark_exe, cwd=args.cwd, shell_init=base_runtime_shell_init, timeout=args.timeout)
                    confirm_rows.extend(rows)
                    log_paths.append(str(confirm_log))
                    benchmark_commands.append(shell_join(command))
                if confirm_streamk_entries:
                    rows, commands = run_entries_with_streamk_example(confirm_streamk_entries, logs_dir, args.streamk_example_exe, cwd=args.cwd, shell_init=base_runtime_shell_init, timeout=args.timeout)
                    confirm_rows.extend(rows)
                    log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in confirm_streamk_entries)
                    benchmark_commands.extend(commands)
                all_rows.extend(confirm_rows)
    write_results_csv(all_rows, reports_dir / "gemm_profile_results.csv")
    dispatch_table = build_dispatch_table(
        all_rows,
        shapes_doc,
        top_k=top_k,
        confirm_runs=confirm_runs,
        close_call_threshold=args.close_call_threshold,
        candidate_space=candidate_space,
        hw_spec=env_caps.get("hw_reference_spec"),
    )
    write_json(reports_dir / "gemm_dispatch_table.json", dispatch_table)
    write_json(reports_dir / "optimal_dispatch_table.json", dispatch_table)
    if reference_doc is not None:
        write_json(
            reports_dir / "reference_comparison.json",
            build_reference_comparison(dispatch_table, reference_doc),
        )
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
        "selected_kernel_list": str(selected_kernel_list_path),
        "selected_kernel_filter": str(selected_kernel_filter_path),
        "candidate_build_cmake_config": str(candidate_build_cmake_config_path),
        "candidate_build_plan": str(candidate_build_plan_path),
        "candidate_build_summary": str(candidate_build_summary_path),
        "safe_candidates": str(reports_dir / "bmg_safe_candidates.json"),
        "verified_hw_caps": str(verified_hw_caps_path),
        "results_csv": str(reports_dir / "gemm_profile_results.csv"),
        "dispatch_table": str(reports_dir / "gemm_dispatch_table.json"),
        "optimal_dispatch_table": str(reports_dir / "optimal_dispatch_table.json"),
        "reference_doc": str(reference_doc_path) if reference_doc is not None else "",
        "reference_comparison": str(reports_dir / "reference_comparison.json") if reference_doc is not None else "",
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
    parser.add_argument("--reference-json", default="", help="Optional path to reference/oracle JSON for dataset comparison.")
    parser.add_argument("--ali-workbook", default="", help="Optional Ali GEMM performance workbook. When set, workflow derives gemm_target_shapes.json and reference comparison input from the workbook.")
    parser.add_argument("--max-shapes", type=int, default=0, help="Limit target shapes to the first N entries after loading --shapes-json, --ali-workbook, or the default shape set. 0 disables the limit.")
    parser.add_argument("--constraints-json", default="", help="Optional path to safe_search_constraints.json.")
    parser.add_argument("--compiler-profiles-json", default="", help="Optional path to compiler_profiles.json.")
    parser.add_argument("--kernel-catalog-source", choices=["persisted", "generator"], default="persisted", help="Catalog source for Phase B candidates. 'generator' bridges Intel Xe library generation into the benchmark/search catalog but requires a benchmark binary built from the same generated kernels.")
    parser.add_argument("--kernel-catalog-path", default="", help="Optional path to a persisted kernel catalog JSON. Used when --kernel-catalog-source=persisted.")
    parser.add_argument("--compiled-kernel-list", default="", help="Optional newline-delimited compiled kernel list or regex filter file. When set, Phase B only runs benchmark candidates present in this list.")
    parser.add_argument("--cmake-source-dir", default="", help="Optional source directory used in the generated candidate benchmark CMake build plan. Defaults to --cwd or the current directory.")
    parser.add_argument("--benchmark-build-dir", default="", help="Optional build directory used in the generated candidate benchmark CMake build plan. Defaults to <workspace>/build/candidate_benchmarks.")
    parser.add_argument("--googlebenchmark-dir", default="", help="Optional local Google Benchmark source directory injected into the generated CMake build plan as GOOGLEBENCHMARK_DIR to avoid FetchContent downloads.")
    parser.add_argument("--cmake-cxx-compiler", default="", help="Optional CMAKE_CXX_COMPILER value injected into the generated candidate benchmark CMake build plan, e.g. 'icpx' for oneAPI SYCL builds.")
    parser.add_argument("--build-candidate-benchmark", action="store_true", help="Execute the generated candidate benchmark CMake configure/build plan before Phase B runs, then use the built benchmark executable for screening and confirmation.")
    parser.add_argument("--generator-arch", choices=["bmg", "pvc"], default="bmg", help="Intel Xe generator arch used when --kernel-catalog-source=generator.")
    parser.add_argument("--generator-instantiation-level", type=int, default=0, help="Intel Xe generator instantiation level used when --kernel-catalog-source=generator.")
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
