# Manifest and Kernel Generation System

This is a code/kernel generation system that creates a searchable catalog of CUTLASS kernel operations, bridging build-time generation and runtime selection.

## Architecture Overview

**Two-Phase System:**
1. **Build Time (Python)**: `manifest.py` generates C++ initialization code
2. **Runtime (C++)**: Generated code registers operations into a searchable `Manifest`

```
Python Generator → C++ Files → Compiled Library → Runtime Catalog
```

## Key Components

### Python Generator (`manifest.py`)

**Responsibilities:**
- Filter kernels by GPU architecture (SM/Xe), operation type, patterns
- Group operations by kind/architecture/instruction type  
- Generate C++ initialization functions and CMake files

### Generated File Structure
```
build/tools/library/generated/
├── initialize_all.cpp
├── gemm/20/tensorop/cutlass3x_xe20_tensorop_gemm_bf16_*.cpp
└── manifest.cmake
```

### Architecture Naming
| GPU | Prefix | ID | Example |
|-----|--------|----|---------|
| CUDA | `sm` | 70-90 | `sm80` |
| Intel Xe | `xe` | 12,20 | `xe20` |

## Runtime API

### Core Classes

```cpp
// Manifest: Operation catalog
class Manifest {
  Status initialize();
  void append(Operation *op);
  OperationVector const& operations() const;
};

// Operation: Base kernel interface  
class Operation {
  virtual Status can_implement(void const *config, void const *args) const = 0;
  virtual Status run(void const *args, void *workspace, Stream stream) const = 0;
};
```

### Initialization Hierarchy
```cpp
namespace cutlass::library {
  void initialize_all(Manifest &manifest);                    // All operations
  void initialize_all_gemm_operations(Manifest &manifest);    // GEMM only
  void initialize_all_xe20_gemm_operations(Manifest &manifest); // XE20 GEMM
}
```

## Usage Examples

### Basic Usage
```cpp
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

cutlass::library::Manifest manifest;
cutlass::library::initialize_all(manifest);

// Find BF16 GEMM
for (auto& op : manifest.operations()) {
  if (op->description().name.find("bf16") != std::string::npos) {
    // Use operation...
  }
}
```

### Python Integration
```python
# Use extern "C" wrappers for ctypes integration
from ctypes import CDLL
lib = CDLL("libcutlass_gemm_xe20_gemm.so")
# Call exported C functions that wrap C++ manifest APIs
```

**Example Implementation:** See `examples/11_xe20_cutlass_library/` for a complete CMake-based shared library that exports CUTLASS kernels for Python usage via ctypes.

## Common Patterns

### Lazy Initialization
```cpp
class LazyManifest {
  cutlass::library::Manifest manifest_;
  bool initialized_ = false;
public:
  cutlass::library::Manifest& get() {
    if (!initialized_) {
      cutlass::library::initialize_all(manifest_);
      initialized_ = true;
    }
    return manifest_;
  }
};
```

### Operation Caching
```cpp
class OperationCache {
  std::map<std::string, cutlass::library::Operation*> cache_;
public:
  cutlass::library::Operation* find(const std::string& pattern) {
    if (cache_.count(pattern)) return cache_[pattern];
    // Search manifest and cache result...
  }
};
```

## Build Integration

### CMake Configuration
```bash
# Generate for Intel XE20
cmake .. -DCUTLASS_LIBRARY_GENERATOR_ARCHS="20"
ninja cutlass_library
```

### Python Generator
```bash
python3 generator.py --operations=gemm --architectures=20 --build-dir=.
```

## Performance Tips

- **Selective Initialization**: Only initialize needed operation kinds
- **Operation Caching**: Cache frequently used operations
- **Kernel Filtering**: Use build-time filtering to reduce library size
- **Lazy Loading**: Initialize manifest only when needed

## Debugging

```bash
# List generated operations
nm -D libcutlass_gemm_xe20_gemm.so | grep initialize

# Enable Python debug logging
python3 -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

## References

- **Source**: `python/cutlass_library/manifest.py`
- **Headers**: `tools/library/include/cutlass/library/`
- **Generated**: `build/tools/library/generated/`
- **Examples**: 
  - `examples/11_xe20_cutlass_library/` - CMake-based shared library for Python integration
  - `examples/python/cutlass_library/xe20_gemm_bf16.py` - Python test script using ctypes
