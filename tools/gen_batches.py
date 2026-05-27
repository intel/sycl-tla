#!/usr/bin/env python3
"""Generate kernel batches from catalog for screening."""
import sys, json, argparse
sys.path.insert(0, "test/benchmarks")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-kernels", type=int, default=0, help="Limit total (0=all)")
    args = p.parse_args()

    import os; os.makedirs(f"{args.workspace}/builds", exist_ok=True)

    from intel_gemm_profiler.catalog import generated_layered_bmg_kernel_catalog
    from intel_gemm_profiler.constraints import default_constraints

    cons = default_constraints()
    cat = generated_layered_bmg_kernel_catalog(constraints=cons)
    df = {"bf16": "bf16", "f16": "f16"}.get(args.dtype, "bf16")
    all_k = sorted(set(k["kernel_name"] for k in cat["kernels"] if k.get("dtype_family") == df))

    if args.max_kernels > 0:
        all_k = all_k[:args.max_kernels]

    batches = [all_k[i:i+args.batch_size] for i in range(0, len(all_k), args.batch_size)]
    print(f"Total: {len(all_k)} kernels in {len(batches)} batches", file=sys.stderr)

    manifest = {"dtype": args.dtype, "batch_size": args.batch_size, "batches": []}
    for i, batch in enumerate(batches):
        bid = f"batch_{i:04d}"
        mani_f = f"{args.workspace}/builds/{bid}_manifest.txt"
        with open(mani_f, "w") as f:
            for k in batch: f.write(k + "\n")
        gpu = [5, 7][i % 2]
        manifest["batches"].append({"id": bid, "count": len(batch), "gpu": gpu, "manifest": mani_f})

    with open(f"{args.workspace}/batch_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {args.workspace}/batch_manifest.json", file=sys.stderr)

if __name__ == "__main__":
    main()
