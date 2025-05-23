//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class month;

// constexpr month operator+(const month& x, const months& y) noexcept;
//   Returns: month(int{x} + y.count()).
//
// constexpr month operator+(const months& x, const month& y) noexcept;
//   Returns:
//     month{modulo(static_cast<long long>(int{x}) + (y.count() - 1), 12) + 1}
//   where modulo(n, 12) computes the remainder of n divided by 12 using Euclidean division.
//   [Note: Given a divisor of 12, Euclidean division truncates towards negative infinity
//   and always produces a remainder in the range of [0, 11].
//   Assuming no overflow in the signed summation, this operation results in a month
//   holding a value in the range [1, 12] even if !x.ok(). —end note]
//   [Example: February + months{11} == January. —end example]

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename M, typename Ms>
__host__ __device__ constexpr bool testConstexpr()
{
  M m{1};
  Ms offset{4};
  if (m + offset != M{5})
  {
    return false;
  }
  if (offset + m != M{5})
  {
    return false;
  }
  //  Check the example
  if (M{2} + Ms{11} != M{1})
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using month  = cuda::std::chrono::month;
  using months = cuda::std::chrono::months;

  static_assert(noexcept(cuda::std::declval<month>() + cuda::std::declval<months>()));
  static_assert(noexcept(cuda::std::declval<months>() + cuda::std::declval<month>()));

  static_assert(cuda::std::is_same_v<month, decltype(cuda::std::declval<month>() + cuda::std::declval<months>())>);
  static_assert(cuda::std::is_same_v<month, decltype(cuda::std::declval<months>() + cuda::std::declval<month>())>);

  static_assert(testConstexpr<month, months>(), "");

  month my{2};
  for (unsigned i = 0; i <= 15; ++i)
  {
    month m1 = my + months{i};
    month m2 = months{i} + my;
    assert(m1 == m2);
    unsigned exp = i + 2;
    while (exp > 12)
    {
      exp -= 12;
    }
    assert(static_cast<unsigned>(m1) == exp);
    assert(static_cast<unsigned>(m2) == exp);
  }

  return 0;
}
