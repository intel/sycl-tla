#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import argparse
import importlib.util
import json
from pathlib import Path


def load_profiler_module():
    module_path = Path(__file__).with_name("intel_gemm_profiler.py")
    spec = importlib.util.spec_from_file_location("intel_gemm_profiler", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    profiler = load_profiler_module()
    parser = argparse.ArgumentParser(description="Run Phase B constrained search for the Intel GEMM profiler.")
    parser.add_argument("--workspace", required=True, help="Workspace directory for generated files and reports.")
    parser.add_argument("--dtype", choices=sorted(profiler.SEED_KERNELS.keys()), default="bf16")
    parser.add_argument("--probe-mode", choices=["off", "auto", "static", "run"], default="auto")
    parser.add_argument("--benchmark-exe", default="./build/benchmarks/gemm/cutlass_benchmarks_gemm_sycl")
    parser.add_argument("--streamk-example-exe", default="./build/examples/03_bmg_gemm_streamk/03_bmg_gemm_streamk")
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--shell-init", default="")
    parser.add_argument("--constraints-json", default="")
    parser.add_argument("--compiler-profiles-json", default="")
    parser.add_argument("--shapes-json", default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--confirm-runs", type=int, default=3)
    parser.add_argument("--close-call-threshold", type=float, default=3.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workflow_args = profiler.build_parser().parse_args(
        [
            "--workspace",
            args.workspace,
            "--dtype",
            args.dtype,
            "--probe-mode",
            args.probe_mode,
            "--benchmark-exe",
            args.benchmark_exe,
            "--streamk-example-exe",
            args.streamk_example_exe,
            "--top-k",
            str(args.top_k),
            "--confirm-runs",
            str(args.confirm_runs),
            "--close-call-threshold",
            str(args.close_call_threshold),
        ]
        + (["--cwd", args.cwd] if args.cwd else [])
        + (["--shell-init", args.shell_init] if args.shell_init else [])
        + (["--constraints-json", args.constraints_json] if args.constraints_json else [])
        + (["--compiler-profiles-json", args.compiler_profiles_json] if args.compiler_profiles_json else [])
        + (["--shapes-json", args.shapes_json] if args.shapes_json else [])
        + ["--timeout", str(args.timeout)]
        + (["--dry-run"] if args.dry_run else [])
        + (["--skip-run"] if args.skip_run else [])
    )
    outputs = profiler.workflow(workflow_args)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
