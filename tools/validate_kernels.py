#!/usr/bin/env python3
"""Single-Kernel Validator — runs each of 3424 kernels one-by-one on GPU.

Usage:
  python3 validate_kernels.py \\
    --manifest candidate_build_manifest.json \\
    --workspace /path/to/workspace \\
    --shapes shapes.json \\
    --output validated_results.json \\
    --timeout 300 \\
    --checkpoint-interval 50

Output:
  validated_results.json — per-kernel results with:
    {kernel_id, status, avg_tflops, runtime_ms, tile, SG, stages, ILP, mode, dtype}

Progress:
  validate_checkpoint.json — written every 50 kernels, enables resume
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def load_manifest(path):
    """Load candidate_build_manifest.json."""
    with open(path) as f:
        return json.load(f)


def build_kernel_batch_map(manifest, workspace_dir):
    """Build {kernel_id: {batch_id, exe_path}} map from manifest."""
    kernel_map = {}
    base_dir = Path(manifest.get("_workspace_dir", workspace_dir))
    for batch in manifest.get("selected_kernel_batches", []):
        batch_id = batch["batch_id"]
        batch_dir = base_dir / "build" / "candidate_benchmarks" / "candidate_batch_preflight" / batch_id
        exe = batch_dir / "benchmarks" / "gemm" / "cutlass_benchmarks_gemm_sycl"
        for kernel_id in batch.get("selected_kernel_list", []):
            kernel_map[kernel_id] = {
                "batch_id": batch_id,
                "exe_path": str(exe),
            }
    return kernel_map


def load_shapes(path):
    """Load shapes JSON and return list of (shape_id, m, n, k, dtype_a, dtype_c)."""
    with open(path) as f:
        doc = json.load(f)
    shapes = []
    for s in doc.get("shapes", []):
        shapes.append((
            s["shape_id"], s["m"], s["n"], s["k"],
            s["dtype_a"], s.get("dtype_c", "f32"),
            s.get("layout", "rcr"),
        ))
    return shapes


def build_config_file(kernel_id, shapes, output_path):
    """Write a benchmark config file for a single kernel with all shapes."""
    lines = []
    for sid, m, n, k, da, dc, layout in shapes:
        lines.append(
            f"{kernel_id} --bm_name={kernel_id}__{sid}__validate__0 "
            f"--m={m} --n={n} --k={k} --l=1 --alpha=1.0 --beta=0.0"
        )
    Path(output_path).write_text("\n".join(lines) + "\n")


def parse_benchmark_output(text, kernel_id, shape_id):
    """Parse Google Benchmark output (new 50+50 timer format)."""
    for line in text.splitlines():
        if "avg_tflops=" in line:
            tflops_match = re.search(r"avg_tflops=([0-9.]+)", line)
            runtime_match = re.search(r"(?:runtime_trimmed_mean_ms|avg_runtime_ms)=([0-9.]+)", line)
            if tflops_match and runtime_match:
                return float(tflops_match.group(1)), float(runtime_match.group(1))
    return None, None


def run_kernel(exe_path, config_path, log_path, timeout):
    """Run a single kernel benchmark. Returns (returncode, timed_out, stdout_text)."""
    cmd = [exe_path, "--config_file=" + str(config_path)]
    stdout = ""
    timed_out = False

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        # Try best-effort cleanup, but don't block on hung GPU processes
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, _ = proc.communicate(timeout=3)
        except (subprocess.TimeoutExpired, OSError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, _ = proc.communicate(timeout=3)
            except (subprocess.TimeoutExpired, OSError):
                # GPU hung in D-state — cannot kill, move on
                stdout = "TIMEOUT_GPU_HUNG"

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(stdout or "")
    return proc.returncode if not timed_out else 124, timed_out, stdout or ""


def compute_ilp(kernel_id):
    """Extract tile and SG from kernel name, compute ILP."""
    # Pattern: ..._MxNxK_SGSGMxSGN_STSTAGES
    m = re.search(r"_(\d+)x(\d+)x(\d+)_SG(\d+)x(\d+)", kernel_id)
    if m:
        tm, tn, tk, sm, sn = map(int, m.groups())
        return (tm // sm // 8) * (tn // sn // 16)
    return 0


def extract_metadata(kernel_id):
    """Extract (tile_m, tile_n, tile_k, sg_m, sg_n, stages, mode, dtype_family) from kernel_id."""
    m = re.search(r"_(\d+)x(\d+)x(\d+)_SG(\d+)x(\d+)_ST(\d+)", kernel_id)
    if m:
        tm, tn, tk, sm, sn, st = map(int, m.groups())
    else:
        tm = tn = tk = sm = sn = st = 0

    # Mode
    if "_StreamK_" in kernel_id:
        mode = "streamk"
    elif "_DataParallel_" in kernel_id:
        mode = "data_parallel"
    elif "_SplitK_" in kernel_id:
        mode = "splitk"
    elif "_Gemm_" in kernel_id or "_GemmExhaustive_" in kernel_id:
        mode = "gemm"
    else:
        mode = "gemm"

    # Dtype family
    if "BF16" in kernel_id:
        dtype = "bf16"
    elif "FP16" in kernel_id or "F16" in kernel_id:
        dtype = "f16"
    elif "TF32" in kernel_id:
        dtype = "tf32"
    else:
        dtype = "unknown"

    return tm, tn, tk, sm, sn, st, mode, dtype


def load_checkpoint(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {"completed": {}, "results": {}}


def save_checkpoint(path, completed, results):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"completed": completed, "results": results}, f, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Single-Kernel GPU Validator")
    parser.add_argument("--manifest", required=True, help="candidate_build_manifest.json path")
    parser.add_argument("--shapes", required=True, help="Shapes JSON path")
    parser.add_argument("--output", default="validated_results.json", help="Output JSON path")
    parser.add_argument("--checkpoint", default="validate_checkpoint.json", help="Checkpoint file for resume")
    parser.add_argument("--timeout", type=int, default=300, help="Per-kernel timeout in seconds")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Checkpoint every N kernels")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    parser.add_argument("--workspace", default=".", help="Workspace root directory")
    args = parser.parse_args()

    # Load manifest + shapes
    manifest = load_manifest(args.manifest)
    kernel_map = build_kernel_batch_map(manifest, args.workspace)
    shapes = load_shapes(args.shapes)

    print(f"Manifest: {len(kernel_map)} kernels")
    print(f"Shapes: {len(shapes)} shapes")
    print(f"Timeout: {args.timeout}s")

    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint)
    completed = ckpt["completed"]
    results = ckpt["results"]

    # Resume
    # Build a sorted list from the manifest batch order
    batch_order = {}
    for batch in manifest.get("selected_kernel_batches", []):
        for kid in batch.get("selected_kernel_list", []):
            if kid not in batch_order:
                batch_order[kid] = batch["batch_index"]
    remaining = [k for k in sorted(kernel_map.keys(), key=lambda x: batch_order.get(x, 9999)) if k not in completed]
    print(f"Completed: {len(completed)}, Remaining: {len(remaining)}")

    if args.dry_run:
        print("DRY RUN — first 5 kernels:")
        for k in remaining[:5]:
            print(f"  {k} → {kernel_map[k]['exe_path']}")
        return

    if not remaining:
        print("All kernels already validated!")
        return

    workspace = Path(args.workspace)
    configs_dir = workspace / "generated" / "configs"
    logs_dir = workspace / "logs" / "validation"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for i, kernel_id in enumerate(remaining):
        entry = kernel_map[kernel_id]
        exe_path = entry["exe_path"]

        if not Path(exe_path).exists():
            print(f"[{i+1}/{len(remaining)}] SKIP {kernel_id}: exe not found")
            completed[kernel_id] = "skip"
            if (i + 1) % args.checkpoint_interval == 0:
                save_checkpoint(args.checkpoint, completed, results)
            continue

        # Build config
        config_path = configs_dir / f"validate_{kernel_id}.in"
        build_config_file(kernel_id, shapes, config_path)

        # Run
        log_path = logs_dir / f"validate_{kernel_id}.log"
        t0 = time.time()
        rc, timed_out, stdout = run_kernel(exe_path, config_path, log_path, args.timeout)
        elapsed = time.time() - t0

        # Parse results
        status = "hard_timeout" if timed_out else ("pass" if rc == 0 else "fail")
        avg_tflops = 0.0
        avg_runtime_ms = 0.0
        shape_results = {}

        if stdout and not timed_out:
            for sid, _, _, _, _, _, _ in shapes:
                tflops, runtime = parse_benchmark_output(stdout, kernel_id, sid)
                if tflops:
                    shape_results[sid] = {"tflops": tflops, "runtime_ms": runtime}
                    avg_tflops = max(avg_tflops, tflops)  # take best of the shapes
                    avg_runtime_ms = runtime if avg_runtime_ms == 0 else min(avg_runtime_ms, runtime)

        # Extract metadata
        tm, tn, tk, sm, sn, st, mode, dtype = extract_metadata(kernel_id)
        ilp = compute_ilp(kernel_id)

        result = {
            "kernel_id": kernel_id,
            "status": status,
            "avg_tflops": round(avg_tflops, 4),
            "avg_runtime_ms": round(avg_runtime_ms, 6),
            "elapsed_s": round(elapsed, 1),
            "tile_m": tm, "tile_n": tn, "tile_k": tk,
            "sg_m": sm, "sg_n": sn, "stages": st,
            "ilp": ilp, "mode": mode, "dtype": dtype,
            "shape_results": shape_results,
            "batch_id": entry["batch_id"],
        }

        results[kernel_id] = result
        completed[kernel_id] = status

        # Progress
        pct = (len(completed)) / len(kernel_map) * 100
        elapsed_total = time.time() - start_time
        rate = (i + 1) / elapsed_total * 3600 if elapsed_total > 0 else 0
        eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
        print(f"[{i+1}/{len(remaining)}] {kernel_id[:50]:50s} {status:15s} "
              f"tflops={avg_tflops:8.2f} rt={avg_runtime_ms*1000:7.1f}us "
              f"({pct:.1f}% @{rate:.0f}/h ETA {eta:.1f}h)")
        sys.stdout.flush()

        # Checkpoint
        if (i + 1) % args.checkpoint_interval == 0:
            save_checkpoint(args.checkpoint, completed, results)
            print(f"  checkpoint saved ({len(completed)} kernels)")

    # Final save
    save_checkpoint(args.checkpoint, completed, results)

    # Write final results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone! Results: {args.output}")
    print(f"Passed: {sum(1 for r in results.values() if r['status']=='pass')}")
    print(f"Hard timeout: {sum(1 for r in results.values() if r['status']=='hard_timeout')}")
    print(f"Failed: {sum(1 for r in results.values() if r['status']=='fail')}")


if __name__ == "__main__":
    main()
