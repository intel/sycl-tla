## Intel GEMM benchmark-backed Split-K runner plan

### Goal

Bring Split-K into the **benchmark-backed** Phase B search path without reusing `examples/03_bmg_gemm_streamk` as the final best-config backend.

### Current boundary

- **Allowed today**
  - `examples/03_bmg_gemm_streamk` for probe and feature validation
  - Split-K presence in Phase A probe output
- **Not allowed today**
  - using example-backed Split-K rows as final optimal-search evidence
  - emitting example-backed Split-K into the final best-config benchmark search space

### Required implementation

1. **Add a benchmark target with non-legacy Split-K scheduler support**
   - extend the benchmark-side registration path instead of patching legacy code
   - keep the scheduler and argument model aligned with the Xe StreamK-capable kernel path

2. **Register benchmark-visible Split-K kernels**
   - start with `bf16/f16`, `layout=rcr`
   - keep Level 0 scope small and hardware-validated
   - expose kernel ids that match catalog keys

3. **Define runner contract**
   - keep the same subprocess contract used by `cutlass_benchmarks_gemm_sycl`
   - support `--split_k` or equivalent benchmark config field
   - preserve benchmark log parsing shape so current `ResultParser` stays reusable

4. **Gate with Phase A**
   - require probe success before benchmark-backed Split-K kernels enter `bmg_safe_candidates.json`
   - keep `max_split_k=1` fallback when probe evidence is missing or fails

5. **Promote into catalog**
   - add benchmark-backed Split-K entries to `kernel_catalog.json`
   - keep `runner=benchmark`
   - remove the current example-only exception once benchmark-backed coverage exists

### Acceptance criteria

- non-legacy benchmark binary can execute at least one `bf16` and one `f16` Split-K kernel
- profiler can parse those rows through the normal benchmark backend
- `candidate_build_manifest.json` can emit benchmark-backed Split-K variants
- Split-K candidates only enter final dispatch search after Phase A success
- example-backed Split-K remains probe-only until all above items are complete
