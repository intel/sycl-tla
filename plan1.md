# Intel 平台 GEMM profiler 迁移规划

## 问题与目标

基于 `example/task.md`、`example/` 下调研文档，以及 `sycl-tla/tools/profiler`、`sycl-tla/examples` 的现状，规划一个 **对标 CUTLASS `GemmOperationProfiler` 能力** 的 Intel 平台迁移路线。目标不是复刻 CUDA profiler 的全部抽象，而是先在 Intel/BMG/Xe20 路径上建立：

- 固定 shape 的 GEMM 候选搜索；
- 正确性验证；
- 性能 profile；
- best-config 选择；
- shape -> dispatch table 输出；
- 为后续 BlockScaledGemm、oneDNN 对比、测试计划和工程落地提供稳定基线。

## 当前设计纠偏（基于 `example/` 全量文档复核）

经过对 `example/` 目录下分析/设计/实施文档的系统复核，当前实现需要明确纠偏的点有：

1. **自动最优配置搜索是主线，不是附属功能**
   - `task.md`、`0010_design_gap*.md`、`0011_design_on_search_engine.md`、`0012_exe_plan.md`、`0013_tuning_design.md`
     都将其定义为 Phase B / 第一阶段核心交付。
   - 当前代码应继续朝“Phase A 定边界，Phase B 在边界内搜索最优”对齐。

2. **examples 不应用于最优解搜索**
   - 文档目标是 benchmark-backed profiling / search。
   - `examples/03_bmg_gemm_streamk` 只保留在 Phase A probe / 功能验证路径；
     screening / confirm / select_best 必须收敛到 benchmark-backed candidates。

3. **当前实现仍偏 GEMM MVP + seed-kernel 受限搜索**
   - 这满足了垂直切片落地，但还未完全达到文档里 `TileSpaceGenerator -> BenchmarkCodegen -> SearchExecutor -> optimal_dispatch_table`
     的完整形态。
   - 因此后续工作应优先补齐：
     - Phase A / Phase B 入口与产物命名
     - safe candidates 产物
     - optimal dispatch 产物
     - 再逐步扩展真正的 candidate generation。

4. **Phase A / Phase B 产物需与文档命名对齐**
   - 文档主命名是：
     - `verified_hw_caps.json`
     - `safe_search_constraints.json`
     - `compiler_profiles.json`
     - `bmg_safe_candidates.json`
     - `optimal_dispatch_table.json`
   - 当前实现已有多数能力，但输出命名和入口仍需继续对齐。

## 当前执行重点

在保持 non-legacy、benchmark-backed search 边界不退化的前提下，优先完成：

1. **Phase 产物对齐**
   - 让 profiler 直接产出文档约定的 artifact 名称。
2. **Phase 入口对齐**
   - 提供清晰的 Phase A / Phase B 运行入口或等价调用方式。
3. **safe candidates / optimal dispatch 明确化**
   - 避免当前输出语义过于偏 MVP 内部实现。
4. **继续保留现有 B60 可跑通能力**
   - 不破坏 BF16/F16、RCR、Split-K probe 与 benchmark-backed search 的现状。

## 为何“已有不少 SYCL 基础”，但和完整 profiler / 完整搜索空间仍有明显差距

这次对代码的再复核后，结论需要更明确一些：**差距大的根因不是“还没把几个 CUDA API 机械替换掉”，而是当前可跑通路径和最终目标路径在系统边界上并不相同。**

### 1. profiler 的真实缺口是“跨模块语义迁移”，不是单文件 API 翻译

从代码看，`tools/profiler` 里的核心链路是：

```text
options / device_context
  -> device_allocation / tensor init / compare
  -> operation_profiler
  -> gemm_operation_profiler
  -> verification providers / report / workspace save
  -> cutlass_profiler 可执行入口
```

所以虽然很多地方表面上只是：

- `cudaMalloc -> sycl::malloc_device`
- `cudaMemcpy -> queue.memcpy`
- `cudaDeviceSynchronize -> compat::wait`

但真正的工作量来自下面这些“语义耦合”：

1. **设备/流模型不同**
   - profiler 代码不是单个默认 stream 跑到底，而是把 `cudaStream_t`、multi-device、graph capture、计时器、workspace 生命周期串在一起。
   - 即使 `GpuTimer` 已经修好，`operation_profiler.cu` 里的 `predict_iters()`、`profile_kernel_no_cuda_graphs_()`、`profile_kernel_w_cuda_graphs_()` 仍然是围绕 CUDA stream / graph 设计的。

2. **workspace 与验证逻辑深度耦合**
   - `gemm_operation_profiler.cu` 不只是调用 `operation->run()`。
   - 它还负责构造 `GemmWorkspace`、准备 host/device reference、处理 mixed dtype、保存错误 workspace、更新 `verification_map` 和 `Disposition`。
   - 这些逻辑虽然大多是 C++，但它们默认依赖 CUDA provider、device context 和 tensor copy 语义。

3. **provider 体系默认还是 CUDA first**
   - `gemm_operation_profiler.cu` 里 `Provider::kCUBLAS` 仍是实打实的一条主分支，不只是“可选增强”。
   - 即便短期可以只跑 `ReferenceDevice` / `ReferenceHost`，也要把 provider 能力、默认配置、CLI 选项、报表输出一起改顺。

4. **构建系统不是小修**
   - 不是单纯把 `.cu` 改 `.cpp`。
   - 还涉及 generator 产物后缀、manifest 注册、CMake 目标、宏开关、provider 可用性、测试入口、架构过滤和远端 oneAPI 编译路径。

换句话说，**当前 benchmark / examples 的 SYCL 基础，证明的是 kernel 与 benchmark runner 已经能在 Intel GPU 上工作；并不等于 CUDA profiler 这整条工具链已经天然可移植。**

### 2. generator 的差距主要是“覆盖策略保守”，不是“完全没有 Intel 参数化能力”

这里有一个重要纠偏：

- 之前容易把 generator 理解成“SG layout 是自动推导的，所以完全没法控”；
- **但代码实际不是完全如此。**

在 `python/cutlass_library/generator.py` 的 `GenerateXe_TensorOp_16b_DPAS_gemm()` 里，当前 Intel Xe 生成逻辑已经显式维护了一组：

```text
default_tiles_wg_sg = [
  ([wg_tile], [sg_tile]),
  ...
]
```

也就是：

- **tile shape 已显式枚举**
- **SG 布局（准确说 workgroup/subgroup 拆分）也已显式给出一组默认值**
- 还支持 `SYCL_TLA_ADDITIONAL_TILE_SHAPES` 通过 JSON 追加 `wg/sg`

所以真正的问题不是“generator 表达不了 SG layout”，而是：

1. **每个 tile 目前只给一条保守 SG 方案**
   - 这意味着 generator 已经具备“表达能力”，但没有扩成“系统枚举能力”。

2. **stage 仍固定走 auto**
   - 在 `gemm_operation.py` 的 `EmitGemmUniversal3xInstance.emit()` 中，Intel Xe 分支直接走 `StageCountAuto`。
   - 这说明并不是缺少发射通道，而是当前设计选择了“builder 自动决定 stage”，没有把它纳入可搜索维度。

3. **tile scheduler 目前是单一路径**
   - `GenerateXe_TensorOp_16b_DPAS_gemm()` 里调用 `CreateGemmUniversal3xOperator(...)` 时，当前是 `tile_schedulers=[TileSchedulerType.Persistent]`。
   - 也就是说 generator 不是完全没有 scheduler 抽象，而是 Intel path 目前只发一种 persistent/default 风格。

4. **部分 dtype 其实是“代码里有能力，但在统一入口里被显式关掉”**
   - `GenerateIntelXe()` 当前只启用 `GenerateXe_TensorOp_16b_DPAS_gemm()`。
   - `fp8` / `int8` / mixed dtype 的 generation helper 存在，但在统一入口明确被注释禁用。
   - 所以这部分差距不是“没有代码基础”，而是“尚未进入稳定默认构建面”。

这意味着 generator 侧的真实 gap 更准确地说是：

```text
已有: Intel Xe 基础 codegen + 一组安全 tile/wg/sg + auto stage + persistent scheduler
缺少: 面向搜索的系统化枚举、剪枝、分级构建、以及与 benchmark/profiler 候选空间的一致性
```

### 3. “已生成 900+ kernels” 不等于 “当前系统可搜索 900+ kernels”

这是现在感知落差最大的来源。

当前仓库里至少有三层空间：

1. **library generation space**
   - generator/manifest 可以产出很多 CUTLASS library op；
2. **benchmark-backed runnable space**
   - 真正有 benchmark target、能批量跑、能收性能数据的候选集合；
3. **current Python profiler/search space**
   - 目前 `test/benchmarks/intel_gemm_profiler` 实际纳入 catalog / candidates / workflow 的那一小部分。

这三层现在并没有完全打通。

所以即便 `generated_xe20_ops.json` 里已经有上百上千个 op：

- 也不代表它们都已有 benchmark target；
- 不代表都已有稳定命名和结果解析；
- 不代表都能进入当前 Phase B screening / confirm / select_best；
- 更不代表都已经被 probe / microbench 规则裁剪过。

**当前差距大的直接原因，就是“codegen/library 空间”和“真实搜索执行空间”之间还隔着 catalog、build manifest、benchmark codegen、artifact 命名和验证链路。**

### 4. 现阶段路线分叉，反而放大了“差距感”

现在仓库实际上同时存在两条路线：

1. **benchmark-backed Intel search 路线**
   - 已经能在 B60 上跑真实 case、做 dispatch 选择；
2. **完整 CUTLASS `cutlass_profiler` 迁移路线**
   - 目标更全，但还没把 workspace / provider / CLI / report / build 全部迁过来。

第一条路线已经证明“Intel 侧 kernel / benchmark / dataset / 计时 / 选择链路”是可行的；
第二条路线仍然有明显工程量。

因此看起来会像：

- “明明很多能力已经有了”
- “但离完整 profiler 还差不少”

本质上是因为**已经完成的是一条更聚焦、更垂直的替代路径；尚未完成的是通用 CUTLASS profiler 工具链本体。**

## 对当前 gap 评估的修正结论

结合代码现状，当前判断应更新为：

1. **`gpu_timer.cpp` 已不再是主要 gap**
   - timer 基础设施和 SYCL 双路径已经修好。

2. **profiler 的大头工作量不在 `gpu_timer`，而在 `device_allocation + operation_profiler + gemm_operation_profiler + options/main/CMake` 的联动迁移**
   - 这部分仍然是完整 profiler 的主要工期来源。

3. **generator 并不是“只能扩 tile、完全不能扩 SG/stages/scheduler”**
   - SG/workgroup 维度已经有显式入口；
   - 真正缺的是把这些维度系统化、分层化，并与 benchmark/search 执行面打通。

4. **当前最值得优先投入的不是“先把所有 library op 都生成出来”，而是“先把 generator/catalog/benchmark-backed search 的闭环打通并扩大”**
   - 因为它最直接服务 B60 case、也最容易尽快产出最优解数据。

## 本轮文档化交付物（按文件逐个落地到 `design/`）

根据最新反馈，后续不再只保留抽象计划，而是要输出 **逐文件分析文档** 和 **Intel XPU 迁移设计文档**，全部放到仓库的 `design/` 目录下，避免遗漏。

### A. 文档目录结构

计划新增：

```text
design/intel_xpu_profiler/
  00_index.md
  01_gpu_timer_analysis.md
  02_device_allocation_analysis.md
  03_device_context_analysis.md
  04_operation_profiler_analysis.md
  05_gemm_operation_profiler_analysis.md
  06_options_and_entry_analysis.md
  07_cublas_helpers_analysis.md
  08_build_system_analysis.md
  09_generator_search_gap_analysis.md
  10_intel_xpu_migration_design.md
```

如果在实际写作时发现 `cutlass_profiler.cu`、`main.cpp`、`performance_report.cpp`、`problem_space.cpp`、`enumerated_types.cpp` 需要独立展开，也允许拆成更多文件，但最少要覆盖上面这套主干。

### B. 每份“逐文件分析文档”统一模板

每个文件的分析文档都按同一模板写，避免少做：

1. **文件职责**
   - 这个文件在 profiler 链路中的位置；
   - 输入 / 输出 / 与哪些模块交互。

2. **当前 CUDA 依赖逐项清单**
   - API、类型、宏、provider、build 假设；
   - 不是只列函数名，也要列语义依赖（stream、graph、workspace、device context 等）。

3. **现有 SYCL / Intel 可复用基础**
   - 仓库里已经存在的 compat 层、timer、reference、benchmark、generator 能力；
   - 哪些可以直接复用，哪些只能借鉴。

4. **迁移到 Intel XPU 的设计方案**
   - 逐条说明怎么改；
   - 哪些是 API 替换，哪些是架构改写，哪些要删掉。

5. **风险与未决项**
   - 精度、性能、构建复杂度、provider 缺失、命名/产物兼容性。

6. **验证方案**
   - 编译验证；
   - 单元测试/回归；
   - B60/XPU 真机验证。

7. **与其他文件的依赖关系**
   - 明确“必须先做什么，后做什么”，避免后续执行顺序出错。

### C. 逐文件覆盖范围

#### 1. `01_gpu_timer_analysis.md`

覆盖：

- `tools/profiler/include/cutlass/profiler/gpu_timer.h`
- `tools/profiler/src/gpu_timer.cpp`
- `tools/util/include/cutlass/util/sycl_timer.hpp`
- `tools/util/include/cutlass/util/sycl_event_manager.hpp`

重点：

- 说明 `GpuTimer` 已完成哪些修复；
- 解释 event profiling / wall-clock 双路径；
- 说明为什么 timer 已不是主要 gap，但仍然是基础依赖。

#### 2. `02_device_allocation_analysis.md`

覆盖：

- `tools/profiler/src/device_allocation.cu`
- 相关 tensor fill / tensor compare / host-device copy 辅助路径

重点：

- 不只看 `cudaMalloc/cudaMemcpy`；
- 还要拆 `NumericTypeID` 的 switch 分发、host/device tensor 生命周期、compare/save workspace 依赖。

#### 3. `03_device_context_analysis.md`

覆盖：

- `tools/profiler/src/device_context.cu`
- 设备查询、device property、current device / multi-device 管理

重点：

- 说明 `compat::device_count()` 等可复用能力；
- 同时解释为什么它仍需要和 options / profiler 主循环一起改。

#### 4. `04_operation_profiler_analysis.md`

覆盖：

- `tools/profiler/include/cutlass/profiler/operation_profiler.h`
- `tools/profiler/src/operation_profiler.cu`

重点：

- `predict_iters()`
- `profile_kernel_no_cuda_graphs_()`
- `profile_kernel_w_cuda_graphs_()`
- multi-device / stream / graph capture / occupancy 假设

这是 profiler 本体迁移的核心分析文档之一。

#### 5. `05_gemm_operation_profiler_analysis.md`

覆盖：

- `tools/profiler/include/cutlass/profiler/gemm_operation_profiler.h`
- `tools/profiler/src/gemm_operation_profiler.cu`

重点：

- `GemmWorkspace`
- 参数解析与 `ProblemSpace`
- verification provider
- `ReferenceHost` / `ReferenceDevice` / `CUBLAS`
- mixed dtype / workspace save / disposition 更新

这是另一个最大文档，必须逐段拆。

#### 6. `06_options_and_entry_analysis.md`

覆盖：

- `tools/profiler/src/options.cu`
- `tools/profiler/src/cutlass_profiler.cu`
- `tools/profiler/src/main.cpp`

重点：

- CLI 参数；
- device enumeration；
- provider 默认值；
- Intel XPU 下入口行为和用户接口应如何保持兼容。

#### 7. `07_cublas_helpers_analysis.md`

覆盖：

- `tools/profiler/src/cublas_helpers.cu`
- 相关 header / dispatcher 依赖

重点：

- 明确哪些能力短期可以删/关；
- 哪些将来应由 oneMKL 替代；
- 哪些 report / provider 枚举会因此受影响。

#### 8. `08_build_system_analysis.md`

覆盖：

- profiler 相关 `CMakeLists.txt`
- generator 输出后缀、manifest 注册、arch filter、provider 宏

重点：

- `.cu -> .cpp` 只是表面；
- 真正要分析 target、宏、依赖库、selected kernels、oneAPI 编译链如何落地。

#### 9. `09_generator_search_gap_analysis.md`

覆盖：

- `python/cutlass_library/generator.py`
- `python/cutlass_library/gemm_operation.py`
- 以及当前 benchmark-backed profiler catalog / candidate workflow 的连接点

重点：

- 解释 generator 当前已经支持的能力；
- 哪些 dtype/helper 只是被禁用；
- 为什么“generated ops 多”不等于“真实可搜索候选多”；
- catalog / benchmark / profiler 三层空间如何统一。

#### 10. `10_intel_xpu_migration_design.md`

这份是**总设计文档**，不是文件分析重复版，而是把前 1-9 份收敛成一套执行设计：

1. **迁移目标**
   - Intel XPU 上可用的 CUTLASS profiler 最小闭环；
2. **系统边界**
   - benchmark-backed 路线与完整 profiler 路线如何衔接；
3. **阶段划分**
   - 文档产出 -> generator/search 闭环扩展 -> profiler core port -> provider/build 完整化；
4. **模块依赖图**
   - 哪些模块必须先迁；
5. **验证矩阵**
   - 单测 / Python 回归 / oneAPI 语法编译 / B60 真机 case；
6. **交付物**
   - `optimal_dispatch_table`、真实 profiler 可执行、reference comparison、设计文档归档。

### D. 推荐撰写顺序

为了避免少做，文档建议按下面顺序产出：

1. `00_index.md`
2. `01_gpu_timer_analysis.md`
3. `02_device_allocation_analysis.md`
4. `03_device_context_analysis.md`
5. `04_operation_profiler_analysis.md`
6. `05_gemm_operation_profiler_analysis.md`
7. `06_options_and_entry_analysis.md`
8. `07_cublas_helpers_analysis.md`
9. `08_build_system_analysis.md`
10. `09_generator_search_gap_analysis.md`
11. `10_intel_xpu_migration_design.md`

这样会先把 profiler 主链分析完整，再单独收敛 generator/search gap，最后再写总设计，内容不会前后打架。

## 基于 `example/0017_search_engin.md` 的补充结论

`0017_search_engin.md` 明确补齐了一个关键事实：

- **Intel sycl-tla 与 NVIDIA CUTLASS 一样，性能关键 kernel 参数都是编译时固定的**
- 因而剩余搜索系统不能停留在“少量 seed kernels + 运行时选最快”的 MVP 层面
- 后续必须补齐：
  1. **编译时变体生成**
  2. **分层构建策略（Level 0 / 1 / 2）**
  3. **运行时在已编译变体上 sweep 问题形状和少量运行时参数**
  4. **基于 probe / microbench 的剪枝规则**

这与当前 plan 并不冲突，但会改变“剩余实现”的优先级：

- **不回退到完整 CUTLASS `Manifest + Operation` 框架重建**
- 但要引入一个 **更系统化的编译时 variant catalog / instantiation level / benchmark codegen**
- 用它替代当前偏 seed-kernel 的 Phase B 候选来源

## 基于 `example/0018_search_space.md` 的补充结论

`0018_search_space.md` 进一步明确：**搜索空间本身必须由 microbench 结果驱动，而不是只靠经验枚举。**

需要吸收的核心点：

1. **搜索空间拆成两层**

```text
编译时搜索空间:
  tile_m/n/k
  sg_layout
  reg_tiles / ILP class
  dtype
  grf_mode
  （后续可扩到 copy / epilogue）

运行时搜索空间:
  problem_size (m,n,k)
  raster_order
  swizzle_size
  barrier_interval
  k_unroll
  （以及必要时的 split_k runtime knobs）
```

2. **microbench 约束要直接进入搜索空间设计**

目前应吸收进 Phase B 的硬约束 / 启发式包括：

- dpas 形状固定 → `tile_m % 8 == 0`, `tile_n % 16 == 0`, `tile_k % 16 == 0`
- SLM 上限 → 当前按 `<= 64KB` 安全边界建模，后续再由 Phase A probe 实测覆盖
- ILP / reg blocking → 需要显式建模 `reg_tiles` 或等价 ILP class
- `grf_mode=256` 是当前更合理默认
- `barrier_interval` 是 Intel 独有且应纳入搜索/配置层的维度
- `prefetch_strategy` / `k_unroll` 应从单纯 `stages` 中拆出来，不再混成一个维度

3. **当前 plan 中的搜索空间需要纠偏**

之前的 MVP 搜索空间：

```text
tile_m/n/k + sg_m/sg_n + stages + split_k
```

仍然过于粗糙。更新后应逐步收敛到：

```text
Compile-time dimensions:
  tile_m/n/k
  sg_m/sg_n
  reg_tiles / ilp_class
  dtype
  grf_mode
  instantiation_level

Runtime dimensions:
  shape_set
  raster_order
  swizzle_size
  barrier_interval
  k_unroll
```

## 剩余搜索系统实现规划（再次更新）

### R0：先定义“microbench 驱动的搜索空间 schema”

在真正实现 catalog 之前，先锁定一份新的 search-space schema，明确：

- 哪些字段属于 compile-time variant
- 哪些字段属于 runtime sweep
- 哪些字段来自 Phase A probe / microbench 约束
- 哪些字段是 Intel 独有维度（例如 `barrier_interval`, `grf_mode`, `reg_tiles`）

没有这一步，后续 catalog / build manifest / search executor 都会反复返工。

## 剩余搜索系统实现规划（更新后）

### R1：引入分层 kernel catalog / instantiation level

目标：

- 建立 Intel GEMM 的 **Level 0 / 1 / 2** 候选层级
- Level 0 对应当前已验证的少量非 legacy benchmark kernels
- Level 1 扩展到更多 tile / sg layout / dtype 组合
- Level 2 再考虑 stages / copy atom / epilogue 等更大空间

要点：

- 先用结构化 catalog 表达“理论候选空间”
- 再由 Phase A 约束裁剪成 `bmg_safe_candidates.json`
- catalog 必须支持：
  - tile_m / tile_n / tile_k
  - sg_m / sg_n
  - reg_tiles / ilp_class
  - grf_mode
  - split_k
  - dtype / layout
  - runner / benchmark target / compiler profile class
  - instantiation_level
  - runtime defaults / allowed runtime sweep fields (`barrier_interval`, `k_unroll`, `raster_order`, `swizzle_size`)

### R2：把当前 seed-kernel 搜索替换为 catalog-driven candidate generation

目标：

- 当前 `SEED_KERNELS` 更接近 MVP bootstrap
- 后续要切到：

```text
kernel catalog
  -> Phase A constraints
  -> safe candidates
  -> benchmark codegen / build manifest
  -> search executor
```

要求：

- 仍保持 benchmark-backed search 边界
- example 只能继续用于 probe / feature validation
- Split-K 若进入最优解搜索，必须先补 benchmark-backed runner/catalog entry
- candidate generation 需要先应用 microbench / probe 规则：
  - dpas 对齐
  - SLM 上限
  - ILP / reg tile 约束
  - barrier / prefetch / grf_mode 的默认启发式

### R3：补 BenchmarkCodegen / build manifest

目标：

- 不再依赖手工维护少量 Python seed entry
- 生成：
  - build manifest
  - candidate source/config fragments
  - per-level candidate sets

短期实现：

- 先做 **catalog -> candidate manifest -> config emission**
- 不一定一步到位做数百个 `.cpp` AOT 代码生成
- 但接口和 artifact 要按该方向设计，避免再次返工
- build manifest 里要显式区分：
  - compile-time variant key
  - runtime sweep dimensions

### R4：SearchExecutor 扩展为“编译时变体 × 运行时问题”模型

目标：

- 与 `0017_search_engin.md` 对齐：
  - 编译时变体维度
  - 运行时 shape / alpha / beta / 少量 runtime knobs
- 与 `0018_search_space.md` 对齐：
  - 增加 Intel 独有 runtime/config 维度：
    - `barrier_interval`
    - `k_unroll`
    - 后续可能的 `prefetch_strategy`
- 当前实现只覆盖了“固定问题 + 少量预编译候选”
- 后续要支持：
  - 分 instantiation level 搜索
  - 更多 target shape sets
  - 更明确的 search summary / pruning evidence

### R5：Split-K 的下一步边界

当前正确边界：

- Split-K 只在 probe / 功能验证路径中存在
- 不进入最优解 benchmark 搜索

后续若要进入完整搜索，前置条件是：

1. 先补 non-legacy、benchmark-backed 的 Split-K runner
2. 把它纳入 kernel catalog
3. 通过 Phase A probe 验证稳定性
4. 再放进 Phase B safe candidates

## 基于 `0017_search_engin.md` 的执行 backlog（已完成）

### T0：`define-search-runtime-schema`

先锁定搜索系统的数据面，避免后续 catalog / manifest / search executor 重复返工。

输出应明确：

- compile-time variant key
- runtime sweep key
- instantiation level
- pruning rule inputs
- build manifest 字段
- search result / dispatch 需要保留的证据字段

### T1：`define-kernel-catalog-schema`

在 T0 之后定义 Intel GEMM kernel catalog：

- Level 0 / 1 / 2
- dtype / layout / tile / sg / grf_mode
- runner / benchmark target
- compile-time 固定字段
- 允许的 runtime sweep 字段

### T2：`implement-catalog-candidates`

把当前 seed list 替换为：

```text
kernel catalog
  -> Phase A constraints
  -> pruning rules
  -> bmg_safe_candidates.json
```

### T3：`implement-build-manifest`

实现第一版 build manifest / benchmark codegen：

- candidate manifest
- per-level config fragments
- compile-time variant 与 runtime sweep 的拆分

### T4：`wire-search-to-catalog`

把 SearchExecutor 接到 catalog/build manifest，而不是当前手工 seed entry。

要求：

- 保持 BF16/F16 的 non-legacy benchmark 搜索可用
- 不让 example 进入 best-config 搜索
- 保持现有 artifact 兼容

### T5：`plan-benchmark-splitk-runner`

Split-K 仍不直接进入搜索系统；先完成 benchmark-backed runner 的专项设计，再决定纳入 catalog。

### T6：`publish-review-status`

在仓库里新增一份 review 文档，供人工 review：

- 文件路径：`media/docs/cpp/intel_gemm_search_system_review_20260429.md`

- 已实现功能
- 尚未完成的功能
- 剩余搜索系统 backlog
- 当前设计边界（尤其是 benchmark-backed search 与 example-only probe 的边界）

该文档需要提交并推送到 GitHub，便于在代码审阅前统一认知。

以上 T0-T6 已完成，当前代码已经具备：

- `search_runtime_schema.json`
- `kernel_catalog.json`
- `candidate_build_manifest.json`
- catalog-driven `bmg_safe_candidates.json`
- 已发布的 review 文档与 Split-K benchmark runner 计划文档

## 当前执行任务（2026-04-29 下午）

### E1：补齐 F16 large-tile（RCR_6）非 legacy benchmark 覆盖

当前 BF16 已包含 `RCR_6 (256x256x32)`，但 F16 仍缺该 large-tile 变体，导致：

- F16 的 Level 0 catalog 不完整；
- prefill 大 shape 的 F16 搜索空间覆盖弱于 BF16。

本轮直接补：

- `benchmarks/gemm/benchmarks_sycl.hpp` 的 F16 `RCR_6`
- profiler `SEED_KERNELS["f16"]` 对应条目
- 相关测试与 artifact 验证

### E2：增加 profiler `--dry-run` smoke 模式

目标：

- 不再只有 `--skip-run` 这种“纯生成文件不执行”的模式；
- 增加一个 **最小真实执行** 模式，用极小 shape 跑通 benchmark-backed screening 流程；
- 用于校验：
  - benchmark binary 可执行
  - `bm_name` / parser 契约未漂移
  - CSV / dispatch / summary 产物链路连通

约束：

- dry-run 仍保持 non-legacy benchmark-backed 边界；
- 默认关闭 confirmation，优先做 screening smoke；
- Phase A probe 在 dry-run 下收敛到最小开销模式，不额外放大运行成本。

### E3：补单测覆盖 dry-run 与 F16 large-tile

新增覆盖：

- F16 candidate space 含 `tm256_tn256_tk32`
- dry-run shape 集与参数收敛逻辑
- wrapper 对 dry-run 参数的透传

## 本轮完成后的剩余执行项

1. **B60 端到端真实运行**
   - 在远端 B60 节点执行一次非 `--skip-run` 的最小 dry-run / screening run
   - 再执行一次小规模 `screening + confirmation + dispatch` 正式 run

2. **Phase A probe 实质化**
   - 优先补 `dpas_baseline_probe`
   - 再补 `compiler_flags_probe`
   - 让 `compiler_profiles.json` 从静态模板变为 probe-aware

3. **kernel catalog 持久化**
   - 把当前运行时生成的 Level 0 catalog 下沉到仓库内 JSON
   - 让其他工具可不依赖 Python 直接消费

4. **benchmark-backed Split-K**
   - 继续保持 example-only probe 边界
   - 在 benchmark 侧确认调度器与 runner 路径后再纳入正式搜索

## profiler 模块化执行方案（新增）

当前 `test/benchmarks/intel_gemm_profiler.py` 已经承担：

- schema/constants
- catalog 加载与生成
- constraints / probe logic
- subprocess runner / parser
- selector / dispatch
- workflow / CLI

下一轮不建议一次性大拆，而是按 **低风险、可回滚** 的顺序抽离：

1. **第一步：抽 `schemas.py` / `utils.py`**
   - 迁出：
     - `SCHEMA_VERSION`
     - `CSV_FIELDS`
     - `SEARCH_RUNTIME_SCHEMA`
     - `now_iso / ensure_dir / write_json / read_json / shell_join`
   - 这一步不改业务逻辑，只减少主文件噪音。

2. **第二步：抽 `catalog.py`**
   - 迁出：
     - `SEED_KERNELS`
     - catalog JSON 加载/生成
     - `candidate_id_for`
     - `ilp_class`
     - `build_kernel_catalog`
   - 保持对 persisted JSON 的优先读取策略不变。

3. **第三步：抽 `constraints.py`**
   - 迁出：
     - `default_constraints`
     - `apply_static_probe_constraints`
     - `apply_run_probe_constraints`
     - probe shape selection helpers
   - 这样 Phase A/B 的约束逻辑可以单独测试。

4. **第四步：抽 `runner.py`**
   - 迁出：
     - `run_benchmark`
     - `run_entries_with_benchmark`
     - `run_entries_with_streamk_example`
     - log parsing / timeout rows
   - 这是最值得模块化的部分，因为 subprocess / parser / timeout 变更最频繁。

5. **第五步：保留 `workflow.py` 作为最后收口**
   - 等前四步稳定后，再把 workflow / CLI 从主文件中收尾迁出。

拆分原则：

- 每次只拆一个稳定层，不在同一 commit 同时改 probe/search 逻辑；
- 每次拆分后都保持现有 test 名称与 B60 smoke 命令不变；
- repo-side artifact 名称和路径保持兼容。

## 当前代码库状态

### 1. 现有 CUTLASS profiler 是 CUDA/NVIDIA 方案

- `sycl-tla/tools/profiler/` 的入口和主流程仍是标准 CUTLASS profiler：
  - `main.cpp` -> `CutlassProfiler` -> `OperationProfiler::profile_all()`
  - 对 `Manifest` 中的 operation 逐个过滤、初始化、验证、profile、report
- 深度绑定 CUDA/NVIDIA 生态：
  - `options.cu` 依赖 `cudaGetDeviceProperties()`、SM count、L2、compute capability
  - `gpu_timer.cpp`、`device_context.cu`、`device_allocation.cu` 绑定 CUDA runtime
  - `cublas_helpers.cu` / `cudnn_helpers.cpp` 绑定验证 provider
  - `tools/profiler/CMakeLists.txt` 直接链接 `cudart` / `cuda_driver` / `cublas` / `cudnn`

### 2. CUTLASS profiler 真正可复用的是“工作流骨架”

现有 profiler 的核心价值是：

```text
problem -> candidate -> verify -> profile -> select_best -> report
```

也就是：

```text
ProblemSpace iterate
  -> enumerate candidates
  -> initialize configuration/workspace
  -> verify correctness
  -> measure runtime
  -> choose best candidate
  -> append report
```

这部分思想可迁移；但 **`generator.py -> Manifest -> library::Operation` 的绑定方式不应在 Intel 侧照搬**。

### 3. Intel 侧现有资产更适合“候选生成 + 批量编译 + subprocess 运行”

- `sycl-tla/examples/00_bmg_gemm`、`03_bmg_gemm_streamk`、`11_xe20_cutlass_library` 等已具备 Intel BMG/Xe20 GEMM kernel 示例、`can_implement / initialize / run` 路径和 reference 校验基础。
- `tools/util/include/cutlass/util/sycl_timer.hpp`、`sycl_event_manager.hpp` 已给出 SYCL profiling 计时基础。
- 但 Intel kernel variant 当前更接近“手写/有限枚举的几个 `.cpp` 或生成少量候选 binary”，**不是 CUTLASS 那种上千个 template operation 编译进 manifest**。

因此 Intel 侧更合理的落地方式应是：

```text
TileSpaceGenerator
  -> Candidate List
  -> BenchmarkCodegen / batch build
  -> subprocess run binaries
  -> parse output
  -> select_best
  -> emit dispatch table
```

而不是重做一套 `Manifest + Operation` 虚函数分发框架。

## 规划原则

### Plan 1：保留 profiler 工作流骨架，不复刻 CUTLASS 抽象层

保留的部分：

- ProblemSpace 驱动
- candidate 过滤与约束检查
- verify / profile / select_best / report 工作流
- 离线调优、CSV/JSON 结果汇总、dispatch table 输出

不复刻的部分：

- `generator.py -> Manifest -> Operation` 编译时注册体系
- runtime 遍历 1000+ operation 的虚函数调用模型

Intel 侧建议实现：

```text
CUTLASS:
  generator.py -> Manifest -> operation->run()

Intel:
  TileSpaceGenerator -> candidate binaries -> subprocess run -> parse results
```

### Plan 2：MVP 只做 GEMM

一期只对标 `GemmOperationProfiler`，不把 scope 扩到 Attention / GroupedGEMM。

原因：

1. CUTLASS profiler 中真正有“搜索最优配置”逻辑的核心就是 `GemmOperationProfiler`
2. GroupedGemm 的调度模型与普通 GEMM 差异大，不适合并入 MVP
3. Attention 本身不属于 CUTLASS profiler 的现有 operation 范围，更适合作为后续独立专题

阶段划分建议：

- **Phase 1 / MVP**：GEMM
- **Phase 2**：BlockScaledGemm + 更多 layout / dtype
- **Phase 3**：GroupedGemm / Attention（按需求单独评估）

### Plan 3：重定义 Intel 搜索空间

CUDA GEMM profiler 的搜索维度并不能直接照搬到 Intel。需要按 Intel/Xe 原生参数重定义。

#### CUTLASS -> Intel 映射

| CUTLASS 维度 | Intel 对等概念 | 处理方式 |
|---|---|---|
| `cta_m/n/k` | `tile_m/n/k` (WG tile) | 搜索重点 |
| `cluster_m/n/k` | 无直接对等 | 丢弃 |
| `raster_order` | 无稳定对等 | MVP 不纳入 |
| `swizzle_size` | 可能存在 cache-friendly mapping | 先不做主搜索维度，可后置 |
| `stages` | `prefetch_stages` | 搜索 |
| `warps_m/n/k` | `sg_m/sg_n` | 搜索重点 |
| `split_k_mode/slices` | `split_k` | 搜索 |
| `inst_m/n/k` | DPAS atom | 固定，不搜索 |
| — | `block_copy` 形态 | 作为约束/黑名单 |
| — | `SLM` 用量 | 作为约束 |

#### Intel MVP 搜索空间

```text
Search space:
  tile_m   in [8, 16, 32, 64, 128, 256]
  tile_n   in [64, 128, 256]
  tile_k   in [32, 64]
  sg_m     in [1, 2, 4, 8]
  sg_n     in [4, 8]
  stages   in [1, 2, 3]
  split_k  in [1, 2, 4]
```

#### 非搜索约束（由 Phase A 探测）

```text
Constraints:
  DPAS atom            = fixed
  safe SLM limit       = probed
  broken block copies  = blacklisted
  compiler profiles    = probed best set
  unsupported combos   = filtered before build/run
```

### Plan 4：离线调优优先，在线查表后置

迁移路径应保持与 CUTLASS profiler 的实际使用方式一致：

```text
offline tuning
  -> results CSV/JSON
  -> best config selection
  -> dispatch table
  -> online lookup
```

oneDNN 在该体系中的定位：

- **correctness baseline / reference**
- 不是搜索逻辑的一部分
- 性能对比可以放进 report，但不参与 best-config 选择

## 分阶段计划

### Phase A：能力探测与安全边界建立

目标：确定 Intel 平台上哪些候选配置是“值得搜索且可稳定运行”的。

输出：

- `verified_hw_caps.json`
- `compiler_issues.json`
- `compiler_profiles.json`
- `safe_search_constraints.json`

工作项：

1. 探测 DPAS 固定能力边界
2. 探测 block copy 可用/不可用组合
3. 探测 SLM 安全上限
4. 探测 occupancy / subgroup 组合的有效范围
5. 对比 compiler flags / compiler profile，确定最佳编译配置
6. 探测 prefetch stages 的有效区间

结论用途：

- 作为 TileSpaceGenerator 的合法性过滤条件
- 避免把明显坏掉的配置放入后续批量编译和 profile

#### 编译选项的三层结构

Intel 侧编译配置需要拆成三层管理：

1. **前端编译器 flags**
   - 例如 `-fsycl-targets=...`、`-O2/-O3`、profiling/debug 宏开关
2. **IGC 后端环境变量**
   - 例如 `IGC_ExtraOCLOptions`、`IGC_VISAOptions`、`IGC_VectorAliasBBThreshold`
3. **Level Zero / SYCL 运行时编译选项**
   - 例如 `SYCL_PROGRAM_COMPILE_OPTIONS`

这三层选项互相耦合，不能简单视为单一 `CXX_FLAGS`。

#### compiler profile 处理原则

- 不把 compiler 选项放进 Phase B 的逐候选搜索空间
- 而是在 Phase A 用少量代表性 tile 做 compiler profile probe
- 输出 `compiler_profiles.json`
- Phase B 编译候选时，按 tile 特征选择对应 compiler profile

这样避免把候选数乘以 compiler 组合数，导致编译成本失控。

#### compiler_flags_probe

Phase A 增加专门的 `compiler_flags_probe`，目标是确定不同 tile 区间的最优编译配置。

建议至少区分三类：

- `large_tile`
- `medium_tile`
- `small_tile`

probe 关注的核心点：

- 128-GRF vs 256-GRF
- `IGC_VISAOptions=-perfmodel` 是否有稳定收益
- `-O2` 与更激进优化组合的收益是否稳定
- profiling/debug 相关选项在性能模式下是否需要关闭

预期产物：

```text
compiler_profiles.json
  -> large_tile  : env + cmake flags
  -> medium_tile : env + cmake flags
  -> small_tile  : env + cmake flags
```

如果 probe 结果显示所有 tile 区间都收敛到同一组配置，则 Phase B 可退化为全局统一 compiler profile。

### Phase B：GEMM 候选生成与批量运行

目标：建立 Intel 版 `GemmOperationProfiler` 等价工作流。

建议模块：

- `TileSpaceGenerator`
  - 生成 `tile_m/n/k + sg_m/sg_n + stages + split_k` 候选
  - 应用 Phase A 约束过滤
- `BenchmarkCodegen`
  - 采用“源码模板 + 批量编译”模式
  - 为每个候选生成 benchmark 源文件或模板实例
  - 批量编译为 candidate binaries
  - 根据 candidate tile 特征注入对应 compiler profile
  - 统一命名和输出格式
- `BenchmarkRunner`
  - subprocess 执行 binary
  - 收集 correctness / runtime / tflops
- `ResultParser`
  - 解析 stdout/CSV/JSON
  - 汇总到统一结果结构
- `BestSelector`
  - 对同一 shape 的候选执行 confirmation 后再选最优
- `DispatchTableEmitter`
  - 输出 shape -> best-config 映射

#### Phase A / Phase B 解耦策略

Phase B 不应被 Phase A 完全阻塞。建议：

```text
if verified_hw_caps.json exists:
  constraints = load(verified_hw_caps.json)
else:
  constraints = BMG_DEFAULT_CONSTRAINTS
```

即：

- Phase A 完成前，Phase B 可以先基于 hardcoded 的 BMG 安全默认约束启动
- Phase A 完成前，Phase B 也可以先使用 `BMG_DEFAULT_COMPILER_PROFILE`
- Phase A 结果产出后，再替换默认约束并收紧候选空间
- 这样 probe 开发与 GEMM MVP 主链路可以并行推进

#### Target shape set 与 dispatch table key

MVP 中除了 candidate tile space，还需要明确定义 **problem shape space**。

建议与 CUTLASS profiler 保持一致：由输入驱动，而不是自动生成全空间。输入来源分两类：

1. 用户显式输入
   - 单个 shape：`(m, n, k)`
   - shape list：CSV / JSON 文件
2. 预定义 shape set
   - 来自真实业务场景的 GEMM shape 集合
   - 例如 decode / prefill / 常见线性层 shape

MVP 建议的 dispatch table key：

```text
key = (layout, dtype, m, n, k)
```

一期先保留**精确 shape key**，待结果稳定后，再考虑区间/分桶压缩策略。

#### verify 的具体实现路径

MVP 中 verify 建议直接 **内置在 benchmark binary 中**，而不是增加独立 reference runner。

每个 candidate binary 支持：

- `--verify=1`
- `--epsilon=<float>`
- `--m=... --n=... --k=...`

binary 内部执行流程：

```text
1. 运行 candidate kernel -> Computed
2. 运行 reference GEMM -> Reference
3. compare(Computed, Reference, epsilon)
4. 输出 PASS / FAIL / max_error / runtime
```

reference 路径建议：

- MVP 默认：复用 sycl-tla examples 已有的 host/device GEMM reference 逻辑
- Phase 2：增加 oneDNN correctness baseline

也就是说，**MVP 不把 oneDNN 作为唯一 verify 依赖**，先优先复用仓库现有 reference。

#### BestSelector confirmation 机制

Intel 版 BestSelector 建议引入稳定性确认，而不是只做单次 `max`：

```text
1. 对每个 shape，所有 candidates 各跑 1 次 -> 选 top-3
2. top-3 各跑多次 -> 取 median runtime / median tflops
3. 以 median 结果选最终 best
4. 如果 top-1 与 top-2 差距很小，则标记 close_call
```

这一步是 MVP 结果可信度的重要保证。

#### 输出 schema 定义

需要先定义中间产物 schema，再并行开发模块。至少包括：

- `verified_hw_caps.json`
- `safe_search_constraints.json`
- `compiler_profiles.json`
- `gemm_target_shapes.json`
- `gemm_candidate_space.json`
- `gemm_profile_results.csv`
- `gemm_dispatch_table.json`

每个文件需明确：

- 主键/索引字段
- shape 字段表示方式
- compiler profile 的 key / 选择规则表示方式
- layout / dtype / tile / sg / stages / split_k 命名
- correctness / runtime / tflops / close_call 字段
- 必填与可选字段

### Phase C：GEMM MVP 交付

目标：让 Intel 平台具备最小可用的 GEMM 搜索系统。

MVP 范围：

- GEMM only
- 首批聚焦 `RCR` layout
- 首批 dtype 聚焦 `bf16/f16`
- 支持固定 shape 和 shape list 批量输入
- 支持 correctness check
- 支持 runtime/tflops 输出
- 支持 best-config 结果输出

MVP 输出：

- `gemm_candidate_space.json`
- `gemm_profile_results.csv`
- `gemm_dispatch_table.json`
- `gemm_mvp_design.md`（设计说明）
- `gemm_mvp_test_plan.md`（测试说明）

### Phase D：Phase 2 扩展

在 GEMM MVP 稳定后扩展：

1. BlockScaledGemm
2. 更多 layout / dtype
3. oneDNN correctness baseline 标准化接入
4. 启发式缩减搜索空间
5. 再评估 GroupedGemm / Attention 是否值得独立立项

## 推荐任务拆分

### 任务 1：CUTLASS Gemm profiler 拆解

输出：

- `GemmOperationProfiler` 的能力模型
- verify/profile/select_best/report 的流程图
- CUDA 依赖点与 Intel 替换点清单

### 任务 2：Intel GEMM 资产梳理（建议最先启动）

输出：

- 现有 BMG/Xe20 GEMM variants 清单
- 可复用 runner / timer / reference / build 入口清单
- 当前 layout / dtype / scheduler 支持矩阵

### 任务 3：Schema 合同定义

输出：

- `schemas.md`
- 所有 JSON/CSV 中间产物的字段、类型、示例值
- 模块间输入输出契约

说明：

- 该任务应在任务 1 和任务 2 完成后立即执行
- 目的是在并行开发 `TileSpaceGenerator`、`BenchmarkRunner`、`ResultParser`、`BestSelector` 之前先锁定数据契约

### 任务 4：Phase A 约束探测设计

输出：

- probe 列表
- 约束字段定义
- `verified_hw_caps.json` / `safe_search_constraints.json` / `compiler_profiles.json` 数据格式

### 任务 5：GEMM 搜索空间设计

输出：

- 搜索维度定义
- target shape set 定义
- dispatch table key 粒度定义
- 候选裁剪规则
- 候选命名规则
- binary 输出格式与结果格式
- 对已锁定 schema 的字段映射与约束规则补充

### 任务 6：MVP 架构设计

输出：

- `TileSpaceGenerator`
- `BenchmarkCodegen`
- `BenchmarkRunner`
- `ResultParser`
- `BestSelector`
- `DispatchTableEmitter`
- binary 内置 verify 设计
- default constraints 与 probe constraints 切换机制
- compiler profile 选择与注入机制

### 任务 7：验证、测试与 demo 设计

输出：

- oneDNN correctness baseline 方案
- correctness/performance/regression 测试项
- 真实 GEMM 调优 demo 流程

## 建议执行顺序

1. **先启动任务 2：Intel GEMM 资产梳理**
2. 并行快速收尾任务 1：CUTLASS Gemm profiler 拆解
3. 任务 1 + 任务 2 完成后，立即固化任务 3：Schema 合同定义
4. 然后并行推进任务 4：Phase A 约束探测设计 与任务 5：GEMM 搜索空间设计
5. 在 schema、搜索空间和 probe 约束明确后，进入任务 6：MVP 架构设计
6. 最后进入任务 7：验证、测试与 demo 设计

## 初版 todo 列表

- `analyze-cutlass-gemm-profiler`：拆解 `GemmOperationProfiler` 的搜索、验证、报告逻辑
- `inventory-intel-gemm-assets`：梳理 sycl-tla 中可用于 GEMM profiler 迁移的 Intel 资产，优先确认 verify/build/runner 的复用入口
- `lock-data-schemas`：前置定义 JSON/CSV 中间产物 schema 与模块间输入输出契约
- `design-phase-a-probes`：定义硬件能力探测、compiler profile probe 与安全约束输出
- `define-gemm-search-space`：定义 Intel GEMM 搜索空间、target shape set、dispatch key 和候选裁剪规则
- `design-binary-subprocess-workflow`：设计 binary codegen + subprocess runner + result parser 主链路
- `design-gemm-mvp-deliverables`：定义 GEMM MVP 的输出物、dispatch table 和测试方案
- `phase2-blockscaled-roadmap`：规划 MVP 之后的 BlockScaledGemm 扩展路线

### 任务依赖图

```text
inventory-intel-gemm-assets --------\
                                      -> lock-data-schemas -> define-gemm-search-space -> design-binary-subprocess-workflow -> design-gemm-mvp-deliverables
analyze-cutlass-gemm-profiler ------/

design-phase-a-probes --------------/   (可选增强依赖；未完成时先使用 BMG_DEFAULT_CONSTRAINTS / BMG_DEFAULT_COMPILER_PROFILE)

design-gemm-mvp-deliverables -> phase2-blockscaled-roadmap
```

## 关键风险

- 如果试图复刻 `Manifest + Operation`，会把项目拖入不必要的抽象重建
- 如果一期范围扩到 GroupedGemm / Attention，会显著抬高复杂度并稀释主目标
- 如果不先做 Phase A 探测，搜索空间会被 compiler/hardware 不稳定性污染
- 如果把 oneDNN 放进搜索主循环，会让 profiling 与 baseline comparison 耦合过深
- `BenchmarkCodegen` 采用源码模板 + 批量编译后，候选数量较大时编译时间可能成为瓶颈，因此需要支持并行编译、增量编译和失败重试
- Intel 编译配置由前端 flags、IGC 环境变量、Level Zero 运行时选项三层组成；若不先通过 compiler profile probe 收敛，性能结果会难以解释，且候选编译成本会失控

## 结论

本规划的核心结论是：

> **先在 Intel 平台重建 `GemmOperationProfiler` 的能力，而不是复刻 CUTLASS profiler 的全部框架。**

即：

- 保留工作流骨架；
- 改用 `candidate generation + batch build + subprocess run + result parse`；
- MVP 只做 GEMM；
- 先做离线调优与 dispatch table；
- oneDNN 作为 correctness baseline；
- 后续再扩展到 BlockScaledGemm 和其他算子。
