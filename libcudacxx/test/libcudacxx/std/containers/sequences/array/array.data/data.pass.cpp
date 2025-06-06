//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// T *data();

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstddef> // for cuda::std::max_align_t
#include <cuda/std/cstdint>

#include "test_macros.h"

struct NoDefault
{
  __host__ __device__ constexpr NoDefault(int) {}
};

__host__ __device__ constexpr bool tests()
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    C c = {1, 2, 3.5};
    static_assert(noexcept(c.data()));
    T* p = c.data();
    assert(p[0] == 1);
    assert(p[1] == 2);
    assert(p[2] == 3.5);
  }
  {
    typedef double T;
    typedef cuda::std::array<T, 0> C;
    C c = {};
    static_assert(noexcept(c.data()));
    T* p = c.data();
    unused(p);
  }
  {
    typedef double T;
    typedef cuda::std::array<const T, 0> C;
    C c = {{}};
    static_assert(noexcept(c.data()));
    const T* p = c.data();
    unused(p);
    static_assert((cuda::std::is_same<decltype(c.data()), const T*>::value), "");
  }
  {
    typedef NoDefault T;
    typedef cuda::std::array<T, 0> C;
    C c = {};
    static_assert(noexcept(c.data()));
    T* p = c.data();
    unused(p);
  }
  {
    cuda::std::array<int, 5> c = {0, 1, 2, 3, 4};
    assert(c.data() == &c[0]);
    assert(*c.data() == c[0]);
  }

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");

  // Test the alignment of data()
  {
    typedef cuda::std::max_align_t T;
    typedef cuda::std::array<T, 0> C;
    const C c                 = {};
    const T* p                = c.data();
    cuda::std::uintptr_t pint = reinterpret_cast<cuda::std::uintptr_t>(p);
    assert(pint % alignof(T) == 0);
  }
  return 0;
}
