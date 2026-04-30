# Intel GEMM Profiler 迁移与修复状态报告

## 当前结论

当前 `main` 分支已经完成本轮 **Intel/B60 GEMM profiler 迁移纠偏、搜索空间扩展、配置回退修复和测试补强** 的主要工作。  
现阶段代码处于 **可继续做性能迭代，但核心正确性问题已收口** 的状态。

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
- 当前回归结果：**44/44 OK**

### B60 远端

- profiler Python 回归：**44/44 OK**
- benchmark-backed BF16 Phase B 已可跑通
- performance mode 下，B60 大 shape 已观察到显著优于早期小 tile winner 的结果

## 当前仍需继续的工作

### 1. B60 性能稳定性与复测

当前代码逻辑已经收敛，但 B60 多次运行仍存在性能波动。  
下一阶段需要做更受控的 A/B 复测，确认：

- 相同 build hash 下的重复性
- performance mode / 机况 / 频率状态影响
- 最优 kernel 是否稳定落在 `tm64_tn128_tk32_sg4x4` 一类候选上

### 2. 128-GRF experiment 的真实收益验证

`perf_128grf_experiment` 现在只是被正确标注为实验配置，**尚未被证明优于默认 256-GRF baseline**。  
下一步需要在 B60 上做明确的重新编译 A/B：

- baseline：256-GRF + large-register-file
- 去 hint
- 显式 128-GRF

在拿到稳定数据前，应继续保持 `perf_default` 作为默认选择。

### 3. 搜索系统与最初规划文档继续对齐

本轮工作已经把现有 profiler 修到可用和可扩展，但还没有完全达到最初规划中的完整搜索系统形态。  
仍需继续推进：

- Phase A / Phase B artifact 命名对齐
- `safe_candidates` / `optimal_dispatch_table` 产物进一步明确化
- 更系统化的 catalog / instantiation level / build manifest 演进

### 4. 非 RCR layout 的策略决策

当前 profiler 对 unsupported layout 会显式拒绝，这是比静默失败更安全的行为。  
后续需要决定：

- 继续保持显式拒绝并文档化
- 或者补齐 catalog / search 支持，把非 RCR layout 纳入系统

## 当前建议

如果下一阶段目标是 **继续逼近 B60 峰值 TFLOPS**，建议优先顺序如下：

1. 先做 **B60 稳定复测**，确认当前 `tm64_tn128_tk32_sg4x4` 的可重复性  
2. 再做 **128-GRF experiment A/B**，决定是否保留为长期实验分支  
3. 之后再进入 **更大 catalog / 更高 instantiation level** 的扩展

如果下一阶段目标是 **工程化收口**，建议优先顺序如下：

1. 补 Phase A / Phase B artifact 命名对齐  
2. 固化 `safe_candidates` 和 `optimal_dispatch_table` 输出  
3. 再决定 metadata 是否继续保持 advisory-only，还是在 workflow 中增加约束

## 结论

截至当前 `main` 分支，本轮工作已经完成：

- 核心配置架构修正
- B60 搜索空间补洞
- 关键 review bug 修复
- fallback 与 metadata 漂移防护
- 本地与远端回归闭环

项目现在的状态是：**核心 correctness 问题已完成，后续重点转向性能验证和搜索系统继续扩展。**
