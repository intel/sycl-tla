# Runtime environment for B70 profiler
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export IGC_VectorAliasBBThreshold=10000
export ZE_AFFINITY_MASK=7
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
