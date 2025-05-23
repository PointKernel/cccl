//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_unbounded_array<T>
// T is an array type of unknown bound ([dcl.array])

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, bool B>
__host__ __device__ void test_array_imp()
{
  static_assert(B == cuda::std::is_unbounded_array<T>::value, "");
  static_assert(B == cuda::std::is_unbounded_array_v<T>, "");
}

template <class T, bool B>
__host__ __device__ void test_array()
{
  test_array_imp<T, B>();
  test_array_imp<const T, B>();
  test_array_imp<volatile T, B>();
  test_array_imp<const volatile T, B>();
}

typedef char array[3];
typedef char incomplete_array[];

class incomplete_type;

class Empty
{};
union Union
{};

class Abstract
{
  __host__ __device__ virtual ~Abstract() = 0;
};

enum Enum
{
  zero,
  one
};
typedef void (*FunctionPtr)();

int main(int, char**)
{
  //  Non-array types
  test_array<void, false>();
  test_array<cuda::std::nullptr_t, false>();
  test_array<int, false>();
  test_array<double, false>();
  test_array<void*, false>();
  test_array<int&, false>();
  test_array<int&&, false>();
  test_array<Empty, false>();
  test_array<Union, false>();
  test_array<Abstract, false>();
  test_array<Enum, false>();
  test_array<FunctionPtr, false>();

  //  Array types
  test_array<array, false>();
  test_array<incomplete_array, true>();
  test_array<incomplete_type[], true>();

  return 0;
}
