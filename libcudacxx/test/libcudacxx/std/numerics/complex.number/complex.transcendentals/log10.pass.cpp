//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>
//   complex<T>
//   log10(const complex<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(const cuda::std::complex<T>& c, cuda::std::complex<T> x)
{
  assert(log10(c) == x);
}

template <class T>
__host__ __device__ void test()
{
  test(cuda::std::complex<T>(0, 0), cuda::std::complex<T>(-cuda::std::numeric_limits<T>::infinity(), 0));
}

template <class T>
__host__ __device__ void test_edges()
{
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    cuda::std::complex<T> r = log10(testcases[i]);
    cuda::std::complex<T> z = log(testcases[i]) / cuda::std::log(T(10.0));
    if (cuda::std::isnan(real(r)))
    {
      assert(cuda::std::isnan(real(z)));
    }
    else
    {
      assert(real(r) == real(z));
      assert(cuda::std::signbit(real(r)) == cuda::std::signbit(real(z)));
    }
    if (cuda::std::isnan(imag(r)))
    {
      assert(cuda::std::isnan(imag(z)));
    }
    else
    {
      assert(imag(r) == imag(z));
      assert(cuda::std::signbit(imag(r)) == cuda::std::signbit(imag(z)));
    }
  }
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
  test_edges<double>();
#if _LIBCUDACXX_HAS_NVFP16()
  test_edges<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_edges<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  return 0;
}
