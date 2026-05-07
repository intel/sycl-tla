# Intel GEMM Profiler 迁移与修复状态报告

## 当前结论

当前 `main` 分支已经完成本轮 **Intel/BMG GEMM profiler 迁移纠偏、generated benchmark 搜索闭环、Ali workbook 集成、chunked benchmark 执行、确认选择器补强、candidate batch build/routing 和 runtime dispatch lookup** 的主要工作。

现阶段代码处于 **GEMM MVP 主链路已端到端打通，native `tools/profiler/cutlass_profiler` 已完成单 GEMM generated kernel 的 SYCL 正确性+性能闭环，并具备 exact-shape runtime dispatch table lookup/fallback 基础能力** 的状态。

最新全量 Ali workbook generated workflow 已在远端 BMG 节点通过：

- 76 个 BF16 GEMM shapes
- 14 个 generated benchmark candidates
- 1064 条 benchmark rows
- 1064 passed
- 0 failed
- 0 timeout
- 76 个 dispatch entries
- 76 个 Ali reference matches
- 0 missing dispatch

本地 profiler Python 回归当前为 **78/78 OK**。

补充 native C++ `tools/profiler/cutlass_profiler` GEMM smoke 也已在远端 BMG 节点通过：

- generated manifest 使用 `KERNEL_FILTER_FILE` 精确注册 1 个 GEMM operation
- operation：`cutlass3x_xe20_tensorop_gemm_bf16_bf16_f32_f32_f32_128x128x32_1x1x1_0_tnt_align8`
- shape：`m=128, n=128, k=32`
- tensors：`A=bf16:row, B=bf16:column, C=f32:row, D=f32:row`
- verification provider：`reference_host`
- `Disposition: Passed`
- `reference_host: Passed`
- `Status: Success`
- runtime：`0.006636 ms`
- math throughput：`162.951 GFLOP/s`
- output CSV：`/home/intel/tianfeng/cutlas_profile_validation/profiler_gemm_f32d.gemm.csv`

这个 smoke 证明 `tools/profiler` 本体已经可以在 Intel/SYCL 路径上执行 generated GEMM operation、运行 host reference correctness，并写出真实 profile row。之前 `D=bf16` generated kernel 可 profile 但显示 `not_verified`，当前 verified smoke 先选用 `D=f32` kernel，后续若要覆盖 `D=bf16` reference 仍需单独补验证路径。

补充确认阶段 smoke 也已在远端 BMG 节点通过：

- Ali 子集 2 shapes
- `--top-k 3 --confirm-runs 2`
- 40 rows
- screening + confirm stages 均出现
- 40 passed
- 0 failed
- 2 dispatch entries
- `selection_summary.entries_with_confirmation = 2`
- `selection_summary.incomplete_confirmation_entries = 0`
- dispatch evidence 已记录 median runtime、median TFLOPS、stdev、CV、screening rank、runner-up 和 ranked candidates

补充 candidate batch routing smoke 也已在远端 BMG 节点通过：

- 手工 BF16 RCR shape：`m=128, n=128, k=32`
- generator catalog：`--kernel-catalog-source generator --generator-instantiation-level 1`
- compiled-kernel-list 限定 2 个 generated kernels
- `--candidate-build-batch-size 1`
- `--run-candidate-build-preflight`
- `--use-candidate-build-preflight-benchmarks`
- preflight batches: 2
- passed preflight batches: 2
- aggregate candidate build: not run
- screening rows: 2
- passed rows: 2
- failed rows: 0
- dispatch entries: 1
- routing artifacts 已生成：
  - `screening_selected_kernel_batch_000.in`
  - `screening_manifest_selected_kernel_batch_000.json`
  - `screening_selected_kernel_batch_000.log`
  - `screening_selected_kernel_batch_001.in`
  - `screening_manifest_selected_kernel_batch_001.json`
  - `screening_selected_kernel_batch_001.log`

补充 generated level0 StageCountAuto smoke 也已在远端 BMG 节点通过：

- 手工 BF16 RCR shape：`m=128, n=128, k=32`
- generator catalog：`--kernel-catalog-source generator --generator-instantiation-level 0`
- compiled-kernel-list 限定 1 个 StageCountAuto generated kernel
- `--candidate-build-batch-size 1`
- `--run-candidate-build-preflight`
- `--use-candidate-build-preflight-benchmarks`
- preflight batches: 1
- passed preflight batches: 1
- screening rows: 1
- passed rows: 1
- failed rows: 0
- dispatch entries: 1
- selected candidate 使用 `st0`，对应 generator `StageCountAuto`

补充 F16 / 非 RCR generated layout smoke 也已在远端 BMG 节点通过：

- F16 RRR shape：`m=128, n=128, k=32`
- F16 CCR shape：`m=128, n=128, k=32`
- generator catalog：`--kernel-catalog-source generator --generator-instantiation-level 0`
- compiled-kernel-list 每次限定 1 个 StageCountAuto generated kernel
- `--candidate-build-batch-size 1`
- `--run-candidate-build-preflight`
- `--use-candidate-build-preflight-benchmarks`
- RRR selected candidate：`rrr_f16f16f32_tm128_tn128_tk32_sg4x4_st0_sk1`
- CCR selected candidate：`ccr_f16f16f32_tm128_tn128_tk32_sg4x4_st0_sk1`
- RRR / CCR 均为：
  - preflight batches: 1
  - passed preflight batches: 1
  - screening rows: 1
  - passed rows: 1
  - failed rows: 0
  - dispatch entries: 1

补充 generated StreamK 限制现在已显式进入 artifact：

- Intel Xe `GemmUniversal` 当前只允许 `void` 或 `PersistentScheduler`，generated `StreamKScheduler` 会触发 compile-time static_assert
- workflow 继续把 generated StreamK kernels 排除在 `candidate_build_manifest.json` 之外
- `gemm_candidate_space.json` 现在新增：
  - `candidate_coverage`
  - `candidate_exception_summary`
- BF16/RCR level1 artifact smoke：
  - catalog kernels: 1824
  - matched-signature kernels: 152
  - accepted candidates: 28
  - blocked candidates: 20
  - StreamK exceptions: 76
  - exception reason: `intel_xe_generated_streamk_tile_scheduler_unsupported`

## 当前可复现的 GEMM MVP workflow

典型全量 Ali generated benchmark search 命令如下：

```bash
python3 test/benchmarks/intel_gemm_profiler.py \
  --workspace /tmp/ali_generated_full_chunked \
  --ali-workbook /tmp/ali_gemm_perf_v0.1.xlsx \
  --probe-mode off \
  --kernel-catalog-source generator \
  --generator-arch bmg \
  --generator-instantiation-level 1 \
  --cmake-source-dir /path/to/sycl-tla \
  --benchmark-build-dir /path/to/sycl-tla/build-bench-gen-ali-full \
  --googlebenchmark-dir /path/to/googlebenchmark-src \
  --cmake-cxx-compiler icpx \
  --build-candidate-benchmark \
  --benchmark-entry-chunk-size 14 \
  --confirm-runs 0 \
  --shell-init 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1' \
  --timeout 900
```

关键输出位于 `<workspace>/reports/`：

- `kernel_catalog.json`
- `gemm_candidate_space.json`
- `bmg_safe_candidates.json`
- `candidate_build_manifest.json`
- `candidate_build_plan.json`
- `candidate_build_summary.json`
- `gemm_profile_results.csv`
- `gemm_dispatch_table.json`
- `optimal_dispatch_table.json`
- `run_summary.json`
- `phase_a_summary.json`
- `phase_b_summary.json`
- `reference_comparison.json`

大规模 benchmark 建议使用 `--benchmark-entry-chunk-size`。chunked mode 会生成 `screening_partXXX.in`、`screening_manifest_partXXX.json` 和 `screening_partXXX.log`，避免单个大 config 长时间无输出、超时后难以恢复或诊断。

大规模 generated catalog 的构建诊断建议使用 `--candidate-build-batch-size N` 额外生成 `selected_kernel_filter_partXXX.list`、`selected_kernel_batches` metadata 和 `batch_preflight_plans`。默认 screening 仍使用 aggregate benchmark binary；batch filter files 用于更高 instantiation level 下的隔离编译 preflight、失败二分和手动 retry。需要自动执行 batch preflight 时，可加 `--run-candidate-build-preflight`，workflow 会写出 `candidate_build_preflight_summary.json`。如果要直接消费 per-batch build 产物执行 screening/confirmation，可同时开启 `--use-candidate-build-preflight-benchmarks`；该模式要求 preflight 全部通过，并会按 `candidate.kernel_id` 路由到对应 batch executable。

如果需要更可信的最终调优结果，建议打开确认阶段：

```bash
--top-k 3 --confirm-runs 3 --close-call-threshold 3.0
```

确认阶段会对每个 shape 的 top-k candidates 多次运行，并在 dispatch table 的 `evidence` 中记录 median runtime、median TFLOPS、样本数、confirmation 完整性、方差/CV、screening rank、runner-up 信息和 close-call 标记。

## 已完成内容

### 1. 编译时 / 运行时配置拆分完成

- 将 profiler 配置拆成：
  - `build_config_bmg_perf.json`
  - `runtime_config_bmg_perf.json`
- 编译期 flag 与运行期环境变量不再混在同一个 `env` 中。
- benchmark / workflow 调用已切换到 runtime-only 注入方式，避免编译参数泄漏到运行阶段。

对应提交：

- `5b0d1cab` — `refactor: split profiler build runtime config`

### 2. profiler 配置使用方式已文档化

- README 已明确：
  - 默认配置是当前验证过的 baseline
  - 仍支持自定义实验配置
  - 如何切换 build/runtime config 以及重新构建 benchmark

对应提交：

- `a866a8fc` — `docs: describe profiler config usage`

### 3. 128-GRF 配置已改为实验性配置

- `perf_128grf` 已重命名为 `perf_128grf_experiment`
- 该配置不再携带 256-GRF 相关 hint
- 新增 `compile_env_variant_metadata`
- README 已明确该配置 **不是 production baseline**

对应提交：

- `f0972544` — `fix: mark 128grf config experimental`

### 4. B60 搜索空间已扩展并验证收益

- benchmark 注册与 kernel catalog 已加入更大的候选：
  - `64x128x32 sg4x4`
  - `128x128x32 sg4x4`
  - `128x256x32 sg4x4`
- `choose_candidates_for_shape()` 放宽了对大 tile 的过早剪枝
- B60 上大 shape 的 best candidate 已从小 tile 切到 `tm64_tn128_tk32_sg4x4`
- 大 shape 实测性能从约 `29.8 TFLOPS` 提升到约 `46.2 TFLOPS`

对应提交：

- `8728b276` — `feat: expand b60 profiler search space`

### 5. profiler 诊断逻辑已纠偏

- 低效率 warning 不再误报 memory-bound shape
- runtime schema / catalog 已收敛为当前 benchmark 真正支持的运行时维度
- unsupported layout 不再静默失败，而是明确报错

对应提交：

- `c5d94e4c` — `fix: tighten profiler search diagnostics`

### 6. build config fallback 已补齐并锁定

- `_default_build_config()` 已恢复为与 JSON 一致的完整 variant 集合
- fallback path 已有专门测试覆盖
- fallback 的 env 值与 metadata 不仅检查 key，还要求与 repo JSON 完全一致

对应提交：

- `e1016a29` — `fix: align profiler build config fallback`
- `0eda2803` — `test: deepen fallback build config coverage`
- `24890590` — `test: compare fallback config metadata`

## 当前验证状态

### 本地

- `test/python/cutlass/test_intel_gemm_profiler.py`
- 当前回归结果：**65/65 OK**

### BMG 远端

- generated candidate benchmark auto-build 通过
- Ali workbook full generated workflow 通过
- benchmark-backed BF16 Phase B 已可跑通
- chunked screening 全量 76 chunks 已验证
- top-k confirmation smoke 已验证
- reference comparison matched 76/76

## 当前仍需继续的工作

### 1. 结果可信度增强

当前全量 Ali validation 使用 screening-only 证明了链路稳定。正式调优建议继续补：

- 打开 `--confirm-runs` 的全量/子集复测。
- 记录 median runtime / median TFLOPS 后的 winner 稳定性。
- 关注 `close_call=true` 的 shape。
- 将 variance/CV 作为后续 dispatch table 质量门禁或人工 review 信号。

### 2. Phase A probe 深化

已有 `verified_hw_caps.json`、`safe_search_constraints.json` 和 probe mode，但仍需继续增强：

- 更系统的 hardware capability probe。
- compiler profile probe 与 candidate pruning 的更强绑定。
- probe 失败后的降级策略和报告规范。

### 3. 搜索空间继续扩大

当前全量 Ali 跑通的是 BF16、RCR、level 1 generated candidates。后续可继续扩展：

- 更多 dtype/layout。
- 更高 generator instantiation level。
- compile-time variant 和 runtime sweep 的 schema 化边界。
- candidate build 失败隔离和增量复用。

### 4. StreamK 纳入 benchmark-backed search

StreamK example 可用于功能验证，但 generated `_stream_k` kernels 当前因 Intel Xe scheduler specialization 限制被过滤。后续需要在 kernel/generator 层解决该限制，然后再把 Split-K/StreamK 纳入 benchmark-backed search。

### 5. 非 GEMM profiler family 真实 instance 和 correctness

多个 family 已经进入 SYCL build/CLI/dry_run，但仍缺真实 Intel operation instances 和 correctness baseline：

- GroupedGemm
- BlockScaledGemm
- BlockwiseGemm
- RankK / Rank2K
- TRMM / SYMM
- Sparse GEMM
- Conv2d / Conv3d

其中 RankK/Rank2K/TRMM/SYMM 目前在 SYCL reference verification 下显式 `NotSupported`，需要后续补 ReferenceHost/ReferenceDevice 或合适 baseline。

最新 audit 也确认 Intel Xe library generator 当前只生成 GEMM instances；因此这些非 GEMM family 虽然 profiler CLI/build 已纳入，但不能通过 `KERNEL_FILTER_FILE` 直接构建出真实 BMG operation row。下一步若继续推进非 GEMM，需要先在 generator/library 侧补 Intel operation instance，再回到 profiler smoke。

### 6. Runtime dispatch table 集成

当前 `gemm_dispatch_table.json` 和 `optimal_dispatch_table.json` 已能生成，并新增了 Python runtime lookup helper：

- runtime lookup key。
- fallback 策略。
- artifact 版本兼容。

已完成部分：

- lookup key 固定为 `layout, dtype_a, dtype_b, dtype_c, dtype_acc, m, n, k`
- 支持 `gemm_dispatch_table.json` / `optimal_dispatch_table.json` file path 或内存 dict 加载
- 校验 `schema_version`
- 拒绝 duplicate `shape_key`
- exact-shape 命中返回 selected dispatch entry
- shape miss 时返回显式 `missing` 或 `fallback` 结果，包含 fallback reason 和 fallback candidate id

仍未完成的是把该 helper 接入真实推理 runtime 的发布/加载流程。

## 当前建议

下一阶段建议按以下顺序推进：

1. 使用 `--confirm-runs` 做 Ali 子集或全量确认复测，检查 close-call 和 variance。
2. 深化 Phase A probe，将 probe 结果真正反馈到 pruning。
3. 扩展 generated candidate search space。
4. 为非 GEMM family 先补 Intel Xe generator/library operation instances，再做 profiler 真实 profile smoke。
5. 把 runtime dispatch lookup 接到真实推理 runtime 的 artifact 发布/加载流程。
6. 继续推进 StreamK generated benchmark 支持。

## 结论

截至当前 `main` 分支，本轮工作已经完成：

- GEMM MVP generated benchmark search 闭环
- Ali workbook full workflow
- benchmark auto-build
- chunked benchmark execution
- timeout robustness
- confirmation selector evidence
- 多个 tools/profiler family 的 SYCL build/CLI 纳入
- native `tools/profiler/cutlass_profiler` generated GEMM host-reference verified profile row
- runtime dispatch table exact-shape lookup / schema validation / explicit fallback helper
- 本地与远端回归闭环

项目现在的状态是：**GEMM MVP 已能作为离线调优基线使用，native C++ profiler 的 GEMM 主线也已具备最小 verified profile 能力，dispatch artifact 已具备基础 lookup/fallback 能力；后续重点转向 Phase A probe、搜索空间扩展、非 GEMM generator/library instances 和真实 runtime 集成。**
