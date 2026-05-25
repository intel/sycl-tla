#!/usr/bin/env python3
"""Single-Kernel Validator — runs each kernel one-by-one on GPU.

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

GPU monitor & recovery:
  --dmesg-monitor         Check dmesg for GPU errors after each kernel
  --dmesg-snapshot FILE   Save dmesg baseline snapshot (created on start if not exists)
  --reset-on-hang         Auto-reset GPU via xpu-smi on dmesg error detection  
  --reset-delay 10        Seconds to wait after GPU reset before continuing
  --skip-list FILE        File with kernel IDs to skip (one per line)
  --gpu-id N              Set ZE_AFFINITY_MASK=N for this process
  --blacklist-out FILE    Write blacklisted kernel IDs to this file
  --prefilter-tile-m MIN  Skip kernels with tile_m < MIN (small-tile hang prevention)
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


# Pre-built mapping: old-style kernel names (RCR_5, RCR_6, ...) → tile_m
# Extracted from benchmarks_sycl.hpp: using BmgGemm*_RCR_N = Gemm_Bench_*<TileShape_M_N_K, ...>
_OLD_STYLE_TILE_M = {
    "BmgGemmBF16BF16FP32_RCR_5": 8,
    "BmgGemmBF16BF16FP32_RCR_7": 8,
    "BmgGemmBF16BF16FP32_RCR_9": 8,
    "BmgGemmBF16BF16FP32_RCR_16": 16,
    "BmgGemmBF16BF16FP32_RCR_17": 64,
    "BmgGemmBF16BF16FP32_RCR_18": 128,
    "BmgGemmBF16BF16FP32_RCR_19": 128,
    "BmgGemmBF16BF16FP32_RCR_6": 256,
    "BmgGemmFP16FP16FP32_RCR_5": 8,
    "BmgGemmFP16FP16FP32_RCR_7": 8,
    "BmgGemmFP16FP16FP32_RCR_9": 8,
    "BmgGemmFP16FP16FP32_RCR_16": 16,
    "BmgGemmFP16FP16FP32_RCR_17": 64,
    "BmgGemmFP16FP16FP32_RCR_18": 128,
    "BmgGemmFP16FP16FP32_RCR_19": 128,
    "BmgGemmFP16FP16FP32_RCR_6": 256,
}


def extract_metadata(kernel_id):
    """Extract (tile_m, tile_n, tile_k, sg_m, sg_n, stages, mode, dtype_family) from kernel_id."""
    tm = tn = tk = sm = sn = st = 0
    
    # Pattern 1: Exhaustive kernels: _MxNxK_SGsmxsn_STstages
    m = re.search(r"_(\d+)x(\d+)x(\d+)_SG(\d+)x(\d+)_ST(\d+)", kernel_id)
    if m:
        tm, tn, tk, sm, sn, st = map(int, m.groups())
    else:
        # Pattern 2: Source-observed kernels: _Gemm_MxNxK_SGsmxsn (no ST suffix)
        m = re.search(r"_Gemm_(\d+)x(\d+)x(\d+)_SG(\d+)x(\d+)", kernel_id)
        if m:
            tm, tn, tk, sm, sn = map(int, m.groups())
        else:
            # Pattern 3: StreamK/DataParallel/SplitK: _Mode_MxNxK
            m = re.search(r"_(StreamK|DataParallel|SplitK)_(\d+)x(\d+)x(\d+)", kernel_id)
            if m:
                _, tm, tn, tk = m.groups()
                tm, tn, tk = int(tm), int(tn), int(tk)
                sm = sn = 8 if "StreamK" in kernel_id else 4  # default SG for streamk
            else:
                # Pattern 4: Exhaustive without ST: _MxNxK_SGsmxsn (gemm exhaustive)
                m = re.search(r"_(\d+)x(\d+)x(\d+)_SG(\d+)x(\d+)", kernel_id)
                if m:
                    tm, tn, tk, sm, sn = map(int, m.groups())
    
    # Look up old-style kernel tile_m (overrides regex if matched)
    if kernel_id in _OLD_STYLE_TILE_M:
        tm = _OLD_STYLE_TILE_M[kernel_id]

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


# ─── dmesg monitoring & GPU recovery ──────────────────────────────────────────

def _run_cmd(cmd, timeout=10):
    """Run a command, return (returncode, stdout)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return 124, ""
    except Exception:
        return -1, ""


def snapshot_dmesg(path):
    """Save current dmesg as baseline snapshot."""
    _, out = _run_cmd(["dmesg"], timeout=10)
    Path(path).write_text(out)
    return out


def check_dmesg_errors(baseline_snapshot=""):
    """Check for new GPU errors in dmesg since baseline.
    
    Returns (has_error, error_lines) where error_lines is a list of matching lines.
    """
    _, current = _run_cmd(["dmesg"], timeout=10)
    if not current:
        return False, []
    
    # If baseline provided, diff against it
    if baseline_snapshot:
        current_lines = current.splitlines()
        baseline_lines = set(baseline_snapshot.splitlines())
        new_lines = [l for l in current_lines if l not in baseline_lines]
    else:
        new_lines = current.splitlines()
    
    # Patterns that indicate GPU trouble
    error_patterns = [
        r"xe.*error",
        r"xe.*fault",
        r"xe.*timed out",
        r"xe.*hang",
        r"xe.*reset",
        r"xe.*wedged",
        r"xe.*fence.*timed",
        r"i915.*error",
        r"i915.*timed",
        r"i915.*hung",
        r"GPU HANG",
        r"rcs.*reset",
        r"GuC.*error",
        r"guc.*timed",
        r"VM.*fault",
        r"NULL pointer",
        r"kernel BUG",
        r"kernel OOPS",
        r"NMI watchdog",
        r"hard LOCKUP",
    ]
    
    errors = []
    for line in new_lines:
        for pat in error_patterns:
            if re.search(pat, line, re.IGNORECASE):
                errors.append(line)
                break
    
    return len(errors) > 0, errors


def reset_gpu(gpu_id):
    """Reset a GPU via xpu-smi. Returns True on success."""
    print(f"  🔄 Resetting GPU {gpu_id}...")
    rc, out = _run_cmd(["xpu-smi", "config", "-d", str(gpu_id), "--reset"], timeout=60)
    if rc == 0:
        print(f"  ✅ GPU {gpu_id} reset OK")
        return True
    else:
        print(f"  ❌ GPU {gpu_id} reset FAILED: rc={rc} out={out[:200]}")
        return False


def load_skip_list(path):
    """Load kernel IDs to skip from a file."""
    if not path or not Path(path).exists():
        return set()
    return set(l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#"))


def append_blacklist(path, kernel_id):
    """Append a kernel ID to the blacklist file (thread-safe via append)."""
    if not path:
        return
    p = Path(path)
    existing = set()
    if p.exists():
        existing = set(p.read_text().strip().splitlines())
    if kernel_id not in existing:
        with open(p, "a") as f:
            f.write(kernel_id + "\n")


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
    
    # GPU monitoring & recovery
    parser.add_argument("--dmesg-monitor", action="store_true", 
                        help="Check dmesg for GPU errors after each kernel")
    parser.add_argument("--dmesg-snapshot", default="",
                        help="Path to dmesg baseline snapshot file")
    parser.add_argument("--reset-on-hang", action="store_true",
                        help="Auto-reset GPU via xpu-smi when dmesg error detected")
    parser.add_argument("--reset-delay", type=int, default=10,
                        help="Seconds to wait after GPU reset")
    
    # Filtering
    parser.add_argument("--skip-list", default="",
                        help="File with kernel IDs to skip (one per line)")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="Set ZE_AFFINITY_MASK to this GPU device ID")
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="Round-robin partition: this GPU's index (0-based)")
    parser.add_argument("--gpu-total", type=int, default=8,
                        help="Total GPUs for round-robin partition")
    parser.add_argument("--blacklist-out", default="",
                        help="Write blacklisted kernel IDs to this file")
    parser.add_argument("--prefilter-tile-m", type=int, default=0,
                        help="Skip kernels with tile_m < this value (0=no filter)")
    
    args = parser.parse_args()

    # Set GPU affinity
    if args.gpu_id is not None:
        os.environ["ZE_AFFINITY_MASK"] = str(args.gpu_id)
        print(f"ZE_AFFINITY_MASK={args.gpu_id}")

    # Load manifest + shapes
    manifest = load_manifest(args.manifest)
    kernel_map = build_kernel_batch_map(manifest, args.workspace)
    shapes = load_shapes(args.shapes)

    # --- Performance-critical flags (compile + runtime) ---
    os.environ.setdefault("IGC_ExtraOCLOptions", "-cl-intel-256-GRF-per-thread")
    os.environ.setdefault("IGC_VectorAliasBBThreshold", "10000")
    os.environ.setdefault("SYCL_PROGRAM_COMPILE_OPTIONS", "-ze-opt-large-register-file -gline-tables-only")
    os.environ.setdefault("CC", "icx")
    os.environ.setdefault("CXX", "icpx")
    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
    print(f"Flags: 256GRF large-register-file VectorAliasBB=10000")
    print(f"Manifest: {len(kernel_map)} kernels")
    print(f"Shapes: {len(shapes)} shapes")
    print(f"Timeout: {args.timeout}s")

    # Load skip list
    skip_set = load_skip_list(args.skip_list)
    if skip_set:
        print(f"Skip list: {len(skip_set)} kernels will be skipped")

    # Dmesg baseline
    dmesg_baseline = ""
    if args.dmesg_monitor:
        snap_path = args.dmesg_snapshot or "dmesg_baseline.txt"
        if Path(snap_path).exists():
            dmesg_baseline = Path(snap_path).read_text()
            print(f"Dmesg baseline loaded from {snap_path} ({len(dmesg_baseline.splitlines())} lines)")
        else:
            dmesg_baseline = snapshot_dmesg(snap_path)
            print(f"Dmesg baseline saved to {snap_path} ({len(dmesg_baseline.splitlines())} lines)")

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
    
    # Build candidate list with pre-filtering
    remaining_raw = [k for k in sorted(kernel_map.keys(), key=lambda x: batch_order.get(x, 9999)) if k not in completed]
    
    # Apply pre-filters
    prefilter_skipped = 0
    remaining = []
    for k in remaining_raw:
        # Skip list filter
        if k in skip_set:
            completed[k] = "skip_list"
            prefilter_skipped += 1
            continue
        
        # Small-tile filter
        if args.prefilter_tile_m > 0:
            tm, _, _, _, _, _, _, _ = extract_metadata(k)
            if 0 < tm < args.prefilter_tile_m:
                completed[k] = "prefilter_tile_m_small"
                prefilter_skipped += 1
                continue
        
        # Round-robin partition
        if args.gpu_index is not None:
            idx = hash(k) % args.gpu_total
            if idx != args.gpu_index:
                completed[k] = "round_robin_other_gpu"
                prefilter_skipped += 1
                continue
        
        remaining.append(k)
    
    if prefilter_skipped:
        rr_info = f" + round-robin" if args.gpu_index is not None else ""
        print(f"Pre-filter skipped: {prefilter_skipped} kernels (tile_m<{args.prefilter_tile_m} or skip list{rr_info})")
    
    print(f"Completed: {len(completed)}, Remaining: {len(remaining)}")

    if args.dry_run:
        print("DRY RUN — first 10 kernels:")
        for k in remaining[:10]:
            tm, tn, tk, sm, sn, st, mode, dtype = extract_metadata(k)
            print(f"  {k[:60]:60s} tile={tm}x{tn}x{tk} SG={sm}x{sn} st={st} {mode} {dtype}")
        return

    if not remaining:
        print("All kernels already validated!")
        return

    workspace = Path(args.workspace)
    configs_dir = workspace / "generated" / "configs"
    logs_dir = workspace / "logs" / "validation"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Tracking
    hang_count = 0
    consecutive_hangs = 0
    total_resets = 0
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

        # Build config with single shape for faster turnaround
        config_path = configs_dir / f"validate_{kernel_id}.in"
        build_config_file(kernel_id, shapes[:1], config_path)  # use first shape only for validation

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
            for sid, _, _, _, _, _, _ in shapes[:1]:
                tflops, runtime = parse_benchmark_output(stdout, kernel_id, sid)
                if tflops:
                    shape_results[sid] = {"tflops": tflops, "runtime_ms": runtime}
                    avg_tflops = max(avg_tflops, tflops)
                    avg_runtime_ms = runtime if avg_runtime_ms == 0 else min(avg_runtime_ms, runtime)

        # ── Dmesg monitoring ──
        dmesg_error = False
        if args.dmesg_monitor:
            has_err, err_lines = check_dmesg_errors(dmesg_baseline)
            if has_err:
                dmesg_error = True
                print(f"  ⚠️  DMESG ERRORS detected ({len(err_lines)} lines):")
                for line in err_lines[:5]:
                    print(f"     {line[:200]}")
                if len(err_lines) > 5:
                    print(f"     ... and {len(err_lines) - 5} more")
                
                # Blacklist this kernel
                if status not in ("hard_timeout", "pass"):
                    append_blacklist(args.blacklist_out, kernel_id)
                    print(f"  🚫 Blacklisted: {kernel_id}")
                
                # Update baseline to avoid re-triggering on same errors
                dmesg_baseline = snapshot_dmesg(args.dmesg_snapshot)

                # GPU reset
                if args.reset_on_hang and args.gpu_id is not None:
                    if reset_gpu(args.gpu_id):
                        total_resets += 1
                        print(f"  ⏳ Waiting {args.reset_delay}s for GPU recovery...")
                        time.sleep(args.reset_delay)
                        # Refresh dmesg baseline after reset
                        dmesg_baseline = snapshot_dmesg(args.dmesg_snapshot)

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
            "dmesg_error": dmesg_error,
        }

        results[kernel_id] = result
        completed[kernel_id] = status

        # Hang tracking
        if timed_out:
            hang_count += 1
            consecutive_hangs += 1
        else:
            consecutive_hangs = 0

        # If too many consecutive hangs, GPU might be wedged
        if consecutive_hangs >= 3:
            print(f"  ⚠️  {consecutive_hangs} consecutive hangs! GPU might be wedged.")
            if args.reset_on_hang and args.gpu_id is not None:
                if reset_gpu(args.gpu_id):
                    total_resets += 1
                    time.sleep(args.reset_delay)
                    dmesg_baseline = snapshot_dmesg(args.dmesg_snapshot)
                    consecutive_hangs = 0

        # Progress
        pct = (len(completed)) / len(kernel_map) * 100
        elapsed_total = time.time() - start_time
        rate = (i + 1) / elapsed_total * 3600 if elapsed_total > 0 else 0
        remaining_count = len(remaining) - i - 1
        eta = remaining_count / rate if rate > 0 else 0
        status_icon = "⏱️" if timed_out else ("⚠️" if dmesg_error else ("✅" if status == "pass" else "❌"))
        print(f"[{i+1}/{len(remaining)}] {status_icon} {kernel_id[:45]:45s} {status:15s} "
              f"tflops={avg_tflops:8.2f} rt={avg_runtime_ms*1000:7.1f}us "
              f"({pct:.1f}% @{rate:.0f}/h ETA {eta:.1f}h "
              f"hangs:{hang_count} resets:{total_resets})")
        sys.stdout.flush()

        # Checkpoint
        if (i + 1) % args.checkpoint_interval == 0:
            save_checkpoint(args.checkpoint, completed, results)
            print(f"  💾 checkpoint saved ({len(completed)} kernels, {hang_count} hangs, {total_resets} resets)")

    # Final save
    save_checkpoint(args.checkpoint, completed, results)

    # Write final results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Done! Results: {args.output}")
    print(f"Passed: {sum(1 for r in results.values() if r['status']=='pass')}")
    print(f"Hard timeout: {sum(1 for r in results.values() if r['status']=='hard_timeout')}")
    print(f"Failed: {sum(1 for r in results.values() if r['status']=='fail')}")
    print(f"Pre-filter skipped: {prefilter_skipped}")
    print(f"Total hangs: {hang_count}")
    print(f"Total GPU resets: {total_resets}")
    print(f"Elapsed: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
