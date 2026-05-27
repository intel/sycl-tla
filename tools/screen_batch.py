#!/usr/bin/env python3
"""Screen GEMM kernels one-by-one on a single GPU."""
import sys, os, subprocess, argparse, time

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--m", type=int, default=8192)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=1536)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-id", default="")
    args = p.parse_args()

    with open(args.manifest) as f:
        kernels = [l.strip() for l in f if l.strip()]

    env = os.environ.copy()
    env["ZE_AFFINITY_MASK"] = str(args.gpu)
    
    passed = 0
    with open(args.output, "w") as out:
        out.write("kernel,tflops,status,gpu\n")
        for i, kernel in enumerate(kernels):
            try:
                result = subprocess.run(
                    [args.binary, "--kernel", kernel, "--m", str(args.m),
                     "--n", str(args.n), "--k", str(args.k)],
                    capture_output=True, text=True, timeout=args.timeout, env=env,
                )
                output = result.stdout + result.stderr
                tflops = "0"
                status = "FAIL"
                for line in output.split("\n"):
                    if "median_tflops=" in line:
                        tflops = line.split("median_tflops=")[1].split()[0]
                    if "STATUS=OK" in line:
                        status = "OK"
                if status == "OK" and tflops != "0":
                    passed += 1
            except subprocess.TimeoutExpired:
                tflops = "0"
                status = "TIMEOUT"
            out.write(f"{kernel},{tflops},{status},{args.gpu}\n")
            out.flush()
            
            if (i + 1) % 10 == 0:
                print(f"[GPU{args.gpu}] [{args.batch_id}] {i+1}/{len(kernels)} done, {passed} passed", flush=True)

    print(f"[GPU{args.gpu}] [{args.batch_id}] DONE: {passed}/{len(kernels)} passed", flush=True)

if __name__ == "__main__":
    main()
