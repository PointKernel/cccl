//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month;

// constexpr year_month operator+(const year_month& ym, const years& dy) noexcept;
// Returns: (ym.year() + dy) / ym.month().
//
// constexpr year_month operator+(const years& dy, const year_month& ym) noexcept;
// Returns: ym + dy.
//
// constexpr year_month operator+(const year_month& ym, const months& dm) noexcept;
// Returns: A year_month value z such that z - ym == dm.
// Complexity: O(1) with respect to the value of dm.
//
// constexpr year_month operator+(const months& dm, const year_month& ym) noexcept;
// Returns: ym + dm.

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool testConstexprYears(cuda::std::chrono::year_month ym)
{
  cuda::std::chrono::years offset{23};
  if (static_cast<int>((ym).year()) != 1)
  {
    return false;
  }
  if (static_cast<int>((ym + offset).year()) != 24)
  {
    return false;
  }
  if (static_cast<int>((offset + ym).year()) != 24)
  {
    return false;
  }
  return true;
}

__host__ __device__ constexpr bool testConstexprMonths(cuda::std::chrono::year_month ym)
{
  cuda::std::chrono::months offset{6};
  if (static_cast<unsigned>((ym).month()) != 1)
  {
    return false;
  }
  if (static_cast<unsigned>((ym + offset).month()) != 7)
  {
    return false;
  }
  if (static_cast<unsigned>((offset + ym).month()) != 7)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year       = cuda::std::chrono::year;
  using years      = cuda::std::chrono::years;
  using month      = cuda::std::chrono::month;
  using months     = cuda::std::chrono::months;
  using year_month = cuda::std::chrono::year_month;

  auto constexpr January = cuda::std::chrono::January;

  { // year_month + years
    static_assert(noexcept(cuda::std::declval<year_month>() + cuda::std::declval<years>()));
    static_assert(noexcept(cuda::std::declval<years>() + cuda::std::declval<year_month>()));

    static_assert(
      cuda::std::is_same_v<year_month, decltype(cuda::std::declval<year_month>() + cuda::std::declval<years>())>);
    static_assert(
      cuda::std::is_same_v<year_month, decltype(cuda::std::declval<years>() + cuda::std::declval<year_month>())>);

    static_assert(testConstexprYears(year_month{year{1}, month{1}}), "");

    year_month ym{year{1234}, January};
    for (int i = 0; i <= 10; ++i)
    {
      year_month ym1 = ym + years{i};
      year_month ym2 = years{i} + ym;
      assert(static_cast<int>(ym1.year()) == i + 1234);
      assert(static_cast<int>(ym2.year()) == i + 1234);
      assert(ym1.month() == January);
      assert(ym2.month() == January);
      assert(ym1 == ym2);
    }
  }

  { // year_month + months
    static_assert(noexcept(cuda::std::declval<year_month>() + cuda::std::declval<months>()));
    static_assert(noexcept(cuda::std::declval<months>() + cuda::std::declval<year_month>()));

    static_assert(
      cuda::std::is_same_v<year_month, decltype(cuda::std::declval<year_month>() + cuda::std::declval<months>())>);
    static_assert(
      cuda::std::is_same_v<year_month, decltype(cuda::std::declval<months>() + cuda::std::declval<year_month>())>);

    static_assert(testConstexprMonths(year_month{year{1}, month{1}}), "");

    year_month ym{year{1234}, January};
    for (int i = 0; i <= 10; ++i) // TODO test wrap-around
    {
      year_month ym1 = ym + months{i};
      year_month ym2 = months{i} + ym;
      assert(static_cast<int>(ym1.year()) == 1234);
      assert(static_cast<int>(ym2.year()) == 1234);
      assert(ym1.month() == month(1 + i));
      assert(ym2.month() == month(1 + i));
      assert(ym1 == ym2);
    }
  }

  return 0;
}
