//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_day_last;

// constexpr chrono::day day() const noexcept;
//  Returns: wd_

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year                = cuda::std::chrono::year;
  using month               = cuda::std::chrono::month;
  using day                 = cuda::std::chrono::day;
  using month_day_last      = cuda::std::chrono::month_day_last;
  using year_month_day_last = cuda::std::chrono::year_month_day_last;

  static_assert(noexcept(cuda::std::declval<const year_month_day_last>().day()));
  static_assert(cuda::std::is_same_v<day, decltype(cuda::std::declval<const year_month_day_last>().day())>);

  //  Some months have a 31st
  static_assert(year_month_day_last{year{2020}, month_day_last{month{1}}}.day() == day{31}, "");
  static_assert(year_month_day_last{year{2020}, month_day_last{month{2}}}.day() == day{29}, "");
  static_assert(year_month_day_last{year{2020}, month_day_last{month{3}}}.day() == day{31}, "");
  static_assert(year_month_day_last{year{2020}, month_day_last{month{4}}}.day() == day{30}, "");
  static_assert(year_month_day_last{year{2020}, month_day_last{month{5}}}.day() == day{31}, "");
  static_assert(year_month_day_last{year{2020}, month_day_last{month{6}}}.day() == day{30}, "");
  static_assert(year_month_day_last{year{2020}, month_day_last{month{7}}}.day() == day{31}, "");
  static_assert(year_month_day_last{year{2020}, month_day_last{month{8}}}.day() == day{31}, "");
  static_assert(year_month_day_last{year{2020}, month_day_last{month{9}}}.day() == day{30}, "");
  static_assert(year_month_day_last{year{2020}, month_day_last{month{10}}}.day() == day{31}, "");
  static_assert(year_month_day_last{year{2020}, month_day_last{month{11}}}.day() == day{30}, "");
  static_assert(year_month_day_last{year{2020}, month_day_last{month{12}}}.day() == day{31}, "");

  assert((year_month_day_last{year{2019}, month_day_last{month{2}}}.day() == day{28}));
  assert((year_month_day_last{year{2020}, month_day_last{month{2}}}.day() == day{29}));
  assert((year_month_day_last{year{2021}, month_day_last{month{2}}}.day() == day{28}));

  return 0;
}
