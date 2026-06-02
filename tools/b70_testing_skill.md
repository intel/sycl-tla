# B70 GEMM Profiler Testing Skill — Persistent Reference

## 1. Remote Machines

| Name | IP | User/Pass | Notes |
|------|-----|-----------|-------|
| Maginfra2 | 10.239.11.149 | root / intel@123 | B70, 4 GPUs (0-3), 2500MHz |
| Maginfra1 | 192.168.11.21 | root / (key) | Alternate, via foton jump |

**Connect from local:**
```python
import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("10.239.11.149", username="root", password="intel@123", timeout=15)
```

## 2. Critical Paths (Remote)

```
/root/cutlass_profile_device7_b70_2500mhz/
├── sycl-tla/                        # Git repo (push from local)
│   ├── tools/run_seq.sh             # Sequential screening script
│   ├── tools/gen_mini_hpp.py        # Per-batch HPP generator
│   ├── tools/gen_main.py            # main.cpp generator
│   ├── tools/remote_full_retest.sh  # 3-phase validation
│   ├── test/benchmarks/intel_gemm_profiler/
│   │   ├── catalog.py               # Kernel enumeration
│   │   ├── constraints.py           # B70 constraints
│   │   └── source_templates.py      # SG layouts
│   └── benchmarks/gemm/
│       ├── benchmarks_sycl.hpp      # Master HPP (1,773-line)
│       └── main.cpp
├── screen_ws_v3/                    # WORKSPACE (fresh)
│   ├── manifest.json                # Batch manifest
│   ├── builds/batch_XXXX.txt        # Per-batch kernel lists
│   └── results/batch_XXXX_gpuN.csv  # Screening output
├── screen_ws/                       # Old v2 workspace
└── ali_one_8192_4096_1536_.../build/ # Dependencies (GB lib, CUTLASS lib)
```

## 3. Quick Start — Full Screening

```bash
# On remote (Maginfra2):
cd /root/cutlass_profile_device7_b70_2500mhz

# Sync code
cd sycl-tla && git fetch origin && git reset --hard origin/main

# Generate manifest
python3 -c "
import sys, json
sys.path.insert(0, 'sycl-tla/test/benchmarks')
sys.path.insert(0, 'sycl-tla/python')
from intel_gemm_profiler.catalog import generated_layered_bmg_kernel_catalog
from intel_gemm_profiler.constraints import default_constraints
cons = default_constraints()
cat = generated_layered_bmg_kernel_catalog(constraints=cons)
df = 'bf16'
all_k = sorted(set(k['kernel_name'] for k in cat['kernels'] if k.get('dtype_family') == df))
all_k = [k for k in all_k if not k.startswith('03_bmg') and 'streamk_example' not in k]
batches = [all_k[i:i+2] for i in range(0, len(all_k), 2)]
tot = len(batches)
manifest = {'total': len(all_k), 'batch_size': 2, 'batches': []}
for i, batch in enumerate(batches):
    bid = f'batch_{i:04d}'
    mf = f'screen_ws_v3/builds/{bid}.txt'
    with open(mf, 'w') as f:
        for k in batch: f.write(k + '\n')
    manifest['batches'].append({'id': bid, 'count': len(batch), 'gpu': i % 4, 'manifest': mf})
with open('screen_ws_v3/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'{tot} batches ({len(all_k)} kernels)')
"

# Run screening (background, survives disconnect)
RESULTS_DIR=/root/cutlass_profile_device7_b70_2500mhz/screen_ws_v3/results \
  nohup bash sycl-tla/tools/remote_full_retest.sh phase2 \
  > /tmp/retest_v3.log 2>&1 &

# Monitor: tail -f /tmp/retest_v3.log
```

## 4. Check Progress

```bash
# On remote:
tail -20 /tmp/retest_v3.log                           # recent output
ls screen_ws_v3/results/ | wc -l                       # CSVs so far
grep -c "✅" /tmp/retest_v3.log                        # successful batches
grep "COMPILE FAIL\|LINK FAIL" /tmp/retest_v3.log | wc -l  # failures
ps aux | grep phase | grep -v grep                      # is it alive?
```

## 5. Sync Results to Local

```bash
# From local: scp all CSVs
scp root@10.239.11.149:/root/cutlass_profile_device7_b70_2500mhz/screen_ws_v3/results/*.csv /mnt/c/work/src/cutlas_profile/screening_archive/v3_results/

# Merge and analyze (local):
cd /mnt/c/work/src/cutlas_profile/screening_archive/v3_results
head -1 $(ls *.csv | head -1) > all_raw.csv
for f in *.csv; do tail -n +2 "$f" >> all_raw.csv; done
```

## 6. Top 10 Bugs & Fixes

| # | Symptom | Root Cause | Fix |
|---|---------|------------|-----|
| 1 | 300+ kernel redefine errors | Preamble contains BMG_DECLARE_* macro calls that conflict with batch declares | gen_mini_hpp.py: strip ALL BMG_DECLARE_\w+\( lines from preamble (regex was `BMG_DECLARE_\(` — only matched `BMG_DECLARE_(...)`, missed `BMG_DECLARE_GEMM_TILE(...)` |
| 2 | .o file missing after link fail | .DELETE_ON_ERROR in cmake build.make deletes .o when linking GB stub | sed -i '/^\.DELETE_ON_ERROR/d' build.make before make |
| 3 | SG8×8 compile failure | SG8×8 product=64 exceeds B70 max subgroup=32 | constraints.py valid_subgroup_sizes=[16,32]; catalog.py hard-guard default |
| 4 | SplitK hangs B70 GPU | splits=2 causes hardware hang (never returns) | gemm_configuration_sycl.hpp: splits=1 |
| 5 | run_direct() ignores split_k | Missing set_scheduler_splits() call | benchmark_runner.hpp: add set_scheduler_splits() |
| 6 | Perf flags not baked at make | SYCL_PROGRAM_COMPILE_OPTIONS set after cmake configure | Export before make, NOT in cmake |
| 7 | Compile ~4 TFLOPS vs expected ~155 | IGC defaults without -cl-intel-256-GRF-per-thread | Always set: SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only" IGC_VectorAliasBBThreshold=10000 IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread" |
| 8 | StreamK K=32 tiles missing | EXPANDED_STREAMK_TILES only had 16 shapes | Added 13 more K=32 shapes (29 total) |
| 9 | RRR layout skipping StreamK/DP/SplitK | Only Gemm_ had RRR, exhaustive didn't | Added layout="rrr" to exhaustive functions |
| 10 | gen_mini HPP overwrites original | shutil.copy2(output, FULL) at line 132 | run_seq.sh restores via git checkout before each iteration |

## 7. Essential IGC Flags (ALWAYS REQUIRED)

```bash
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_VectorAliasBBThreshold=10000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
# GPU frequency
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $gov 2>/dev/null
done
for gpu in 0 1 2 3; do
    echo 2500 > /sys/class/drm/card${gpu}/gt_max_freq_mhz 2>/dev/null
done
```

Without these: ~4 TFLOPS. With them: 150-163 TFLOPS.

## 8. Build Pipeline Architecture

```
catalog.py              → 1,772 kernel names
gen_mini_hpp.py         → per-batch benchmarks_sycl.hpp (2 kernels each)
gen_main.py             → per-batch main.cpp
cmake make              → .o per batch (33s, j128)
icpx link               → binary per batch
binary --kernel=X       → screen on GPU (15s/kernel)
```

## 9. gen_mini_hpp.py Key Logic

- **classify()**: regex parses kernel name → type (ge/gs/sk/hw), prefix, params
- **covered()**: checks if StreamK/DP/SplitK type already exists in full HPP
- **cfg_atom()**: maps prefix to config name + atom type
- **Preamble stripping**: keeps `#define BMG_DECLARE_*` definitions, removes ALL invocations
- **Key bug**: regex was `^(BMG_DECLARE_|...)(` — missed `BMG_DECLARE_GEMM_TILE(` because `_` after `DECLARE_` wasn\'t matched by `\w`. Fixed: `^(BMG_DECLARE_\w+|...)\(`

## 10. Performance Data Reference

| Metric | Value |
|--------|-------|
| Peak B70 (BF16→F32, 8192×4096×1536) | 163.0 TFLOPS (RRR_Gemm_128x128x32_SG2x4) |
| RRR mean | 90.5 TFLOPS (+11.6% vs RCR) |
| RCR mean | 81.1 TFLOPS |
| ALI peak | ~148 TFLOPS (same problem size) |
| Best scheduler at 256×256×32 SG8×4 | All within 1.4%: Gemm 136.5, DP 144.3, SplitK 144.2, StreamK 142.3 |
| Occupancy sweet spot | 512-2047 (max 155.0, mean 96.9) |

## 11. Common Troubleshooting

| Symptom | Check |
|---------|-------|
| Compile fails (300+ errors) | Check gen_mini_hpp.py preamble strip regex |
| Binary runs at ~4 TFLOPS | Verify IGC flags exported before make |
| .o missing after link | Check .DELETE_ON_ERROR removed |
| SplitK never returns | Set splits=1 in gemm_configuration_sycl.hpp |
| kernel not found error | Check covered() — EXPANDED_STREAMK_TILES set |
| GPU hangs | Check SplitK splits, GPU frequency |
| Manifest 0 batches | Regenerate with latest catalog code |
| Git fetch fails | Use git reset --hard origin/main instead of pull |
