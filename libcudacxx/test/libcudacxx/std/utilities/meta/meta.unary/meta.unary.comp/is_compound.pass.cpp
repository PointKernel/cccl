//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_compound

#include <cuda/std/cstddef> // for cuda::std::nullptr_t
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_compound()
{
  static_assert(cuda::std::is_compound<T>::value, "");
  static_assert(cuda::std::is_compound<const T>::value, "");
  static_assert(cuda::std::is_compound<volatile T>::value, "");
  static_assert(cuda::std::is_compound<const volatile T>::value, "");
  static_assert(cuda::std::is_compound_v<T>, "");
  static_assert(cuda::std::is_compound_v<const T>, "");
  static_assert(cuda::std::is_compound_v<volatile T>, "");
  static_assert(cuda::std::is_compound_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_compound()
{
  static_assert(!cuda::std::is_compound<T>::value, "");
  static_assert(!cuda::std::is_compound<const T>::value, "");
  static_assert(!cuda::std::is_compound<volatile T>::value, "");
  static_assert(!cuda::std::is_compound<const volatile T>::value, "");
  static_assert(!cuda::std::is_compound_v<T>, "");
  static_assert(!cuda::std::is_compound_v<const T>, "");
  static_assert(!cuda::std::is_compound_v<volatile T>, "");
  static_assert(!cuda::std::is_compound_v<const volatile T>, "");
}

class incomplete_type;

class Empty
{};

class NotEmpty
{
  __host__ __device__ virtual ~NotEmpty();
};

union Union
{};

struct bit_zero
{
  int : 0;
};

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
  test_is_compound<char[3]>();
  test_is_compound<char[]>();
  test_is_compound<void*>();
  test_is_compound<FunctionPtr>();
  test_is_compound<int&>();
  test_is_compound<int&&>();
  test_is_compound<Union>();
  test_is_compound<Empty>();
  test_is_compound<incomplete_type>();
  test_is_compound<bit_zero>();
  test_is_compound<int*>();
  test_is_compound<const int*>();
  test_is_compound<Enum>();
  test_is_compound<NotEmpty>();
  test_is_compound<Abstract>();

  test_is_not_compound<cuda::std::nullptr_t>();
  test_is_not_compound<void>();
  test_is_not_compound<int>();
  test_is_not_compound<double>();

  return 0;
}
