//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <cuda/std/type_traits>

// __lazy_enable_if, __lazy_not, _And and _Or

// Test the libc++ lazy meta-programming helpers in <cuda/std/type_traits>

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class Type>
struct Identity : Type
{};

typedef cuda::std::true_type TrueT;
typedef cuda::std::false_type FalseT;

typedef Identity<TrueT> LazyTrueT;
typedef Identity<FalseT> LazyFalseT;

// A type that cannot be instantiated
template <class T>
struct CannotInst
{
  static_assert(cuda::std::is_same<T, T>::value == false, "");
};

template <int Value>
struct NextInt
{
  typedef NextInt<Value + 1> type;
  static const int value = Value;
};

template <int Value>
const int NextInt<Value>::value;

template <class Type>
struct HasTypeImp
{
  template <class Up, class = typename Up::type>
  __host__ __device__ static TrueT test(int);
  template <class>
  __host__ __device__ static FalseT test(...);

  typedef decltype(test<Type>(0)) type;
};

// A metafunction that returns True if Type has a nested 'type' typedef
// and false otherwise.
template <class Type>
struct HasType : HasTypeImp<Type>::type
{};

__host__ __device__ void LazyNotTest()
{
  {
    typedef cuda::std::_Not<LazyTrueT> NotT;
    static_assert(cuda::std::is_same<typename NotT::type, FalseT>::value, "");
    static_assert(NotT::value == false, "");
  }
  {
    typedef cuda::std::_Not<LazyFalseT> NotT;
    static_assert(cuda::std::is_same<typename NotT::type, TrueT>::value, "");
    static_assert(NotT::value == true, "");
  }
  {
    // Check that CannotInst<int> is not instantiated.
    typedef cuda::std::_Not<CannotInst<int>> NotT;

    static_assert(cuda::std::is_same<NotT, NotT>::value, "");
  }
}

__host__ __device__ void LazyAndTest()
{
  { // Test that it acts as the identity function for a single value
    static_assert(cuda::std::_And<LazyFalseT>::value == false, "");
    static_assert(cuda::std::_And<LazyTrueT>::value == true, "");
  }
  {
    static_assert(cuda::std::_And<LazyTrueT, LazyTrueT>::value == true, "");
    static_assert(cuda::std::_And<LazyTrueT, LazyFalseT>::value == false, "");
    static_assert(cuda::std::_And<LazyFalseT, LazyTrueT>::value == false, "");
    static_assert(cuda::std::_And<LazyFalseT, LazyFalseT>::value == false, "");
  }
  { // Test short circuiting - CannotInst<T> should never be instantiated.
    static_assert(cuda::std::_And<LazyFalseT, CannotInst<int>>::value == false, "");
    static_assert(cuda::std::_And<LazyTrueT, LazyFalseT, CannotInst<int>>::value == false, "");
  }
}

__host__ __device__ void LazyOrTest()
{
  { // Test that it acts as the identity function for a single value
    static_assert(cuda::std::_Or<LazyFalseT>::value == false, "");
    static_assert(cuda::std::_Or<LazyTrueT>::value == true, "");
  }
  {
    static_assert(cuda::std::_Or<LazyTrueT, LazyTrueT>::value == true, "");
    static_assert(cuda::std::_Or<LazyTrueT, LazyFalseT>::value == true, "");
    static_assert(cuda::std::_Or<LazyFalseT, LazyTrueT>::value == true, "");
    static_assert(cuda::std::_Or<LazyFalseT, LazyFalseT>::value == false, "");
  }
  { // Test short circuiting - CannotInst<T> should never be instantiated.
    static_assert(cuda::std::_Or<LazyTrueT, CannotInst<int>>::value == true, "");
    static_assert(cuda::std::_Or<LazyFalseT, LazyTrueT, CannotInst<int>>::value == true, "");
  }
}

int main(int, char**)
{
  LazyNotTest();
  LazyAndTest();
  LazyOrTest();

  return 0;
}
