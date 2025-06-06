//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class ForwardIterator1, class ForwardIterator2>
//   constexpr bool   // constexpr after C++17
//   is_permutation(ForwardIterator1 first1, ForwardIterator1 last1,
//                  ForwardIterator2 first2);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    const int ia[]    = {0};
    const int ib[]    = {0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + 0), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + 0),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + 0))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1))
           == false);
  }
  {
    const int ia[]    = {0};
    const int ib[]    = {1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }

  {
    const int ia[]    = {0, 0};
    const int ib[]    = {0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1))
           == false);
  }
  {
    const int ia[]    = {0, 0};
    const int ib[]    = {0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 0};
    const int ib[]    = {1, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 0};
    const int ib[]    = {1, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 1};
    const int ib[]    = {0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 1};
    const int ib[]    = {0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1))
           == false);
  }
  {
    const int ia[]    = {0, 1};
    const int ib[]    = {1, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
  }
  {
    const int ia[]    = {0, 1};
    const int ib[]    = {1, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {1, 0};
    const int ib[]    = {0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {1, 0};
    const int ib[]    = {0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
  }
  {
    const int ia[]    = {1, 0};
    const int ib[]    = {1, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
  }
  {
    const int ia[]    = {1, 0};
    const int ib[]    = {1, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {1, 1};
    const int ib[]    = {0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {1, 1};
    const int ib[]    = {0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {1, 1};
    const int ib[]    = {1, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {1, 1};
    const int ib[]    = {1, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
  }

  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 0, 2};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 1, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 1, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 1, 2};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 2, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 2, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 2, 2};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 0, 1};
    const int ib[]    = {1, 0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1))
           == false);
  }
  {
    const int ia[]    = {0, 0, 1};
    const int ib[]    = {1, 0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 1, 2};
    const int ib[]    = {1, 0, 2};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1))
           == false);
  }
  {
    const int ia[]    = {0, 1, 2};
    const int ib[]    = {1, 2, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1))
           == false);
  }
  {
    const int ia[]    = {0, 1, 2};
    const int ib[]    = {2, 1, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1))
           == false);
  }
  {
    const int ia[]    = {0, 1, 2};
    const int ib[]    = {2, 0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1))
           == false);
  }
  {
    const int ia[]    = {0, 0, 1};
    const int ib[]    = {1, 0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }
  {
    const int ia[]    = {0, 0, 1};
    const int ib[]    = {1, 0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib + 1),
             forward_iterator<const int*>(ib + sa))
           == false);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1))
           == false);
  }
  {
    const int ia[]    = {0, 1, 2, 3, 0, 5, 6, 2, 4, 4};
    const int ib[]    = {4, 2, 3, 0, 1, 4, 0, 5, 6, 2};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib + 1),
             forward_iterator<const int*>(ib + sa))
           == false);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1))
           == false);
  }
  {
    const int ia[]    = {0, 1, 2, 3, 0, 5, 6, 2, 4, 4};
    const int ib[]    = {4, 2, 3, 0, 1, 4, 0, 5, 6, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), forward_iterator<const int*>(ib))
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa))
           == false);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
