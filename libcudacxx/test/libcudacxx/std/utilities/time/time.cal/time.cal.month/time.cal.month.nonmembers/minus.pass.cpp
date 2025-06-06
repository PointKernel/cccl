//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class month;

// constexpr month operator-(const month& x, const months& y) noexcept;
//   Returns: x + -y.
//
// constexpr months operator-(const month& x, const month& y) noexcept;
//   Returns: If x.ok() == true and y.ok() == true, returns a value m in the range
//   [months{0}, months{11}] satisfying y + m == x.
//   Otherwise the value returned is unspecified.
//   [Example: January - February == months{11}. —end example]

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4307) // potential overflow
TEST_DIAG_SUPPRESS_MSVC(4308) // unsigned/signed comparisons

template <typename M, typename Ms>
__host__ __device__ constexpr bool testConstexpr()
{
  {
    M m{5};
    Ms offset{3};
    if (m - offset != M{2})
    {
      return false;
    }
    if (m - M{2} != offset)
    {
      return false;
    }
  }

  //  Check the example
  if (M{1} - M{2} != Ms{11})
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using month  = cuda::std::chrono::month;
  using months = cuda::std::chrono::months;

  static_assert(noexcept(cuda::std::declval<month>() - cuda::std::declval<months>()));
  static_assert(noexcept(cuda::std::declval<month>() - cuda::std::declval<month>()));

  static_assert(cuda::std::is_same_v<month, decltype(cuda::std::declval<month>() - cuda::std::declval<months>())>);
  static_assert(cuda::std::is_same_v<months, decltype(cuda::std::declval<month>() - cuda::std::declval<month>())>);

  static_assert(testConstexpr<month, months>(), "");

  month m{6};
  for (unsigned i = 1; i <= 12; ++i)
  {
    month m1 = m - months{i};
    //      months off = m - month {i};
    int exp = 6 - i;
    if (exp < 1)
    {
      exp += 12;
    }
    assert(static_cast<unsigned>(m1) == static_cast<unsigned>(exp));
    //          assert(off.count()               == static_cast<unsigned>(exp));
  }

  return 0;
}
