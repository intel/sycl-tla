/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/gpu_generics.h"

#if defined(CUTLASS_ENABLE_SYCL)
#include <cute/util/compat/device.hpp>
#include <cute/util/compat/memory.hpp>
#include <sycl/usm.hpp>
#endif

namespace cutlass {
namespace profiler {

#if defined(CUTLASS_ENABLE_SYCL)

struct cudaDeviceProp {
  char name[256];
  int major;
  int minor;
  int multiProcessorCount;
  int l2CacheSize;
  size_t totalGlobalMem;
  int multiGpuBoardGroupID;
};

constexpr cudaDeviceAttr cudaDevAttrClockRate = 1;
constexpr unsigned int cudaStreamNonBlocking = 1;

namespace detail {

inline std::string lowercase(std::string text) {
  for (char &ch : text) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return text;
}

inline int infer_compute_capability(std::string const &device_name) {
  std::string lower = lowercase(device_name);
  if (lower.find("pvc") != std::string::npos || lower.find("ponte") != std::string::npos) {
    return 12;
  }
  if (lower.find("bmg") != std::string::npos || lower.find("battlemage") != std::string::npos ||
      lower.find("xe2") != std::string::npos) {
    return 20;
  }
  if (lower.find("intel") != std::string::npos) {
    return 20;
  }
  return 0;
}

inline sycl::queue queue_for_device(int device) {
  return *compat::get_device(device).default_queue();
}

inline sycl::queue& current_queue() {
  return *compat::get_current_device().default_queue();
}

inline cudaError_t unknown_error() {
  return cudaErrorUnknown;
}

}  // namespace detail

inline cudaError_t cudaGetDeviceCount(int *count) {
  if (!count) {
    return detail::unknown_error();
  }
  *count = int(compat::device_count());
  return cudaSuccess;
}

inline cudaError_t cudaGetDevice(int *device) {
  if (!device) {
    return detail::unknown_error();
  }
  *device = int(compat::get_current_device_id());
  return cudaSuccess;
}

inline cudaError_t cudaSetDevice(int device) {
  compat::select_device(unsigned(device));
  return cudaSuccess;
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
  if (!prop) {
    return detail::unknown_error();
  }

  auto &dev = compat::get_device(unsigned(device));
  auto info = dev.get_device_info();
  std::memset(prop, 0, sizeof(cudaDeviceProp));
  std::strncpy(prop->name, info.get_name(), sizeof(prop->name) - 1);

  int cc = detail::infer_compute_capability(prop->name);
  prop->major = cc / 10;
  prop->minor = cc % 10;
  prop->multiProcessorCount = info.get_max_compute_units();
  prop->l2CacheSize = int(info.get_global_mem_cache_size());
  prop->totalGlobalMem = info.get_global_mem_size();
  prop->multiGpuBoardGroupID = int(info.get_device_id());

  return cudaSuccess;
}

inline cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device) {
  if (!value) {
    return detail::unknown_error();
  }

  auto info = compat::get_device(unsigned(device)).get_device_info();
  if (attr == cudaDevAttrClockRate) {
    *value = info.get_max_clock_frequency() * 1000;
    return cudaSuccess;
  }
  if (attr == cutlass::cudaDevAttrMultiProcessorCount) {
    *value = info.get_max_compute_units();
    return cudaSuccess;
  }

  *value = 0;
  return detail::unknown_error();
}

inline cudaError_t cudaDeviceSynchronize() {
  compat::get_current_device().queues_wait_and_throw();
  return cudaSuccess;
}

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int) {
  if (!stream) {
    return detail::unknown_error();
  }
  *stream = compat::get_current_device().create_queue(false, true);
  return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  if (!stream) {
    return cudaSuccess;
  }
  for (unsigned int device = 0; device < compat::device_count(); ++device) {
    compat::get_device(device).destroy_queue(stream);
  }
  return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  if (stream) {
    stream->wait();
  } else {
    compat::wait();
  }
  return cudaSuccess;
}

inline cudaError_t cudaMalloc(void **ptr, size_t size) {
  if (!ptr) {
    return detail::unknown_error();
  }
  try {
    *ptr = sycl::malloc_device(size, detail::current_queue());
  } catch (std::exception const&) {
    *ptr = nullptr;
    return detail::unknown_error();
  }
  return *ptr ? cudaSuccess : detail::unknown_error();
}

inline cudaError_t cudaFree(void *ptr) {
  try {
    compat::free(ptr, detail::current_queue());
  } catch (std::exception const&) {
    return detail::unknown_error();
  }
  return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void *dst, void const *src, size_t size, cudaMemcpyKind) {
  try {
    compat::memcpy(dst, src, size, detail::current_queue());
  } catch (std::exception const&) {
    return detail::unknown_error();
  }
  return cudaSuccess;
}

#endif

}  // namespace profiler
}  // namespace cutlass
