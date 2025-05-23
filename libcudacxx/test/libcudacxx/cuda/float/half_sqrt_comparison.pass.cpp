//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc, nvcc-11, nvcc-12.0, nvcc-12.1

#include <cuda/std/cmath>

#include "host_device_comparison.h"

#if !_LIBCUDACXX_HAS_NVFP16()
static_assert(false);
#endif // !_LIBCUDACXX_HAS_NVFP16()

struct func
{
  __host__ __device__ __half operator()(cuda::std::size_t i) const
  {
    auto raw = __half_raw();
    raw.x    = (unsigned short) i;
    return cuda::std::sqrt(__half(raw));
  }
};

void test()
{
  compare_host_device<__half>(func());
}

int main(int argc, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, { test(); })

  return 0;
}
