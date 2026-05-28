#!/usr/bin/env python3
"""
Generate per-batch minimal benchmarks_sycl.hpp containing only needed kernel types.

Works by:
1. Parsing the full benchmarks_sycl.hpp into sections
2. For each requested kernel, finding its definition lines + dependencies  
3. Generating a minimal header with just the needed types

Usage:
  python3 tools/gen_mini_hpp.py --manifest batch_0000.txt --output /path/to/benchmarks_sycl.hpp
"""
import re, sys, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FULL_HPP = REPO / "benchmarks/gemm/benchmarks_sycl.hpp"
DEF_FILE = REPO / "benchmarks/gemm/bmg_gemm_source_tile_sg.def"
CONFIG_HPP = REPO / "benchmarks/gemm/gemm_configuration_sycl.hpp"
CUTLASS_H = REPO / "include/cutlass/cutlass.h"

def parse_full_hpp():
    """Parse benchmarks_sycl.hpp into logical sections."""
    with open(FULL_HPP) as f:
        lines = f.readlines()
    return lines

def find_kernel_def(lines, kernel_name):
    """Find all lines needed to define a kernel type."""
    needed = set()
    
    # Look for explicit type definition
    for i, line in enumerate(lines):
        if kernel_name in line:
            # Type definition: "using KernelName = ..."
            if f"using {kernel_name} = " in line or f"using {kernel_name}_" in line:
                # Include TiledMMA dependency (lines above, up to previous kernel or blank)
                j = i - 1
                while j >= 0:
                    prev = lines[j].strip()
                    if prev.startswith("using Bmg") or prev.startswith("// ") or prev.startswith("CUTLASS_CREATE"):
                        if not prev.startswith("// "):
                            needed.add(j)
                        break
                    if prev and not prev.startswith("#") and not prev.startswith("template"):
                        needed.add(j)
                    j -= 1
                needed.add(i)
                # Also include the CUTLASS_CREATE line
                if i + 1 < len(lines) and "CUTLASS_CREATE_GEMM_BENCHMARK" in lines[i+1]:
                    needed.add(i+1)
            
            # CUTLASS_CREATE line
            if f"CUTLASS_CREATE_GEMM_BENCHMARK({kernel_name})" in line:
                needed.add(i)
    
    return sorted(needed)

def find_kernels_by_prefix(lines, prefix):
    """Find all kernel names matching a prefix in the source."""
    names = set()
    for line in lines:
        m = re.findall(r'CUTLASS_CREATE_GEMM_BENCHMARK\((\w+)\)', line)
        for k in m:
            if k.startswith(prefix):
                names.add(k)
    # Also from BMG_DECLARE_STREAMK_TILE expansion
    for line in lines:
        m = re.findall(r'using (\w+) = .*GemmConfiguration', line)
        for k in m:
            if k.startswith(prefix):
                names.add(k)
    return names

def gen_mini_hpp(kernel_list, output_path):
    """Generate minimal benchmarks_sycl.hpp for given kernels."""
    lines = parse_full_hpp()
    
    # Collect all kernel names (resolve prefixes like "BmgGemmBF16_*")
    resolved = set()
    for k in kernel_list:
        if k.endswith('*'):
            resolved |= find_kernels_by_prefix(lines, k.replace('*', ''))
        else:
            resolved.add(k)
    
    # Build output lines
    out = []
    
    # 1. Copyright + includes (lines up to MMAAtom)
    mmaatom_idx = None
    for i, line in enumerate(lines):
        out.append(line)
        if "using MMAAtom = " in line:
            mmaatom_idx = i
            break
    
    out.append("")
    
    # 2. All template/type definitions between MMAAtom and register_gemm_benchmarks
    reg_idx = None
    for i, line in enumerate(lines):
        if "static void register_gemm_benchmarks()" in line:
            reg_idx = i
            break
    
    # 3. Include template blocks (always needed)
    # Extract the Gemm_Bench template blocks
    for i in range(mmaatom_idx+1, reg_idx):
        l = lines[i].strip()
        # Include all template definitions (lines starting with "template" or "using Gemm_Bench")
        if l.startswith("template <") or "using Gemm_Bench_" in l or "BMG_DECLARE" in l or \
           l.startswith("#define BMG_") or l.startswith("using MMAAtom") or \
           l.startswith('using Scheduler'):
            out.append(lines[i])
    
    out.append("")
    
    # 4. Extract type definitions for each kernel
    kernel_lines = set()
    for k in sorted(resolved):
        def_lines = find_kernel_def(lines, k)
        for idx in def_lines:
            # Include dependency lines (TiledMMA types etc)
            if idx not in kernel_lines:
                # Check backward: include the TiledMMA type defined before this kernel
                kernel_lines.add(idx)
                # Also include dependency: if this kernel references BmgTile_X, include it
                depline = lines[idx]
                for tile_match in re.finditer(r'(BmgTile_\w+|BmgGemm_\w+_Tile\w*)', depline):
                    tile_name = tile_match.group(1)
                    for j in range(max(0, idx-15), idx):
                        if tile_name in lines[j] and ("using " + tile_name in lines[j]):
                            kernel_lines.add(j)
    
    # Write kernel type lines in order
    for idx in sorted(kernel_lines):
        l = lines[idx].rstrip()
        if l.startswith("static void register"):
            continue
        out.append(l)
    
    out.append("")
    
    # 5. Registration function
    out.append("static void register_gemm_benchmarks() {")
    for k in sorted(resolved):
        out.append(f"  CUTLASS_BENCHMARK({k});")
    out.append("}")
    out.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(out))
    
    print(f"Generated {output_path}: {len(resolved)} kernels, {len(out)} lines")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    with open(args.manifest) as f:
        kernels = [l.strip() for l in f if l.strip()]
    gen_mini_hpp(kernels, args.output)
