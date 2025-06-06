//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter, typename OutIter,
//          Predicate<auto, InIter::value_type> Pred, class T>
//   requires OutputIterator<OutIter, InIter::reference>
//         && OutputIterator<OutIter, const T&>
//         && CopyConstructible<Pred>
//   constexpr OutIter      // constexpr after C++17
//   replace_copy_if(InIter first, InIter last, OutIter result, Pred pred, const T& new_value);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

constexpr __host__ __device__ bool equalToTwo(const int v) noexcept
{
  return v == 2;
}

template <class InIter, class OutIter>
constexpr __host__ __device__ void test()
{
  constexpr int N           = 5;
  constexpr int ia[N]       = {0, 1, 2, 3, 4};
  int ib[N + 1]             = {0};
  constexpr int expected[N] = {0, 1, 5, 3, 4};
  OutIter r                 = cuda::std::replace_copy_if(InIter(ia), InIter(ia + N), OutIter(ib), equalToTwo, 5);
  assert(base(r) == ib + N);
  for (int i = 0; i < N; ++i)
  {
    assert(ib[i] == expected[i]);
  }

  for (int i = N; i < N + 1; ++i)
  {
    assert(ib[i] == 0);
  }
}

constexpr __host__ __device__ bool test()
{
  test<cpp17_input_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, int*>();

  test<forward_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<forward_iterator<const int*>, forward_iterator<int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<int*>>();
  test<forward_iterator<const int*>, random_access_iterator<int*>>();
  test<forward_iterator<const int*>, int*>();

  test<bidirectional_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<int*>>();
  test<bidirectional_iterator<const int*>, int*>();

  test<random_access_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<random_access_iterator<const int*>, forward_iterator<int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<int*>>();
  test<random_access_iterator<const int*>, int*>();

  test<const int*, cpp17_output_iterator<int*>>();
  test<const int*, forward_iterator<int*>>();
  test<const int*, bidirectional_iterator<int*>>();
  test<const int*, random_access_iterator<int*>>();
  test<const int*, int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
