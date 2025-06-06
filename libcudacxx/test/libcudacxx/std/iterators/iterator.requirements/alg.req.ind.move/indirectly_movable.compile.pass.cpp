//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class In, class Out>
// concept indirectly_movable;

#include <cuda/std/iterator>

#include "MoveOnly.h"
#include "test_macros.h"

// Can move between pointers.
static_assert(cuda::std::indirectly_movable<int*, int*>, "");
static_assert(cuda::std::indirectly_movable<const int*, int*>, "");
static_assert(!cuda::std::indirectly_movable<int*, const int*>, "");
static_assert(cuda::std::indirectly_movable<const int*, int*>, "");

// Can move from a pointer into an array but arrays aren't considered indirectly movable-from.
#if !TEST_COMPILER(MSVC) || TEST_STD_VER != 2017
static_assert(cuda::std::indirectly_movable<int*, int[2]>, "");
#endif // !TEST_COMPILER(MSVC) || TEST_STD_VER != 2017
static_assert(!cuda::std::indirectly_movable<int[2], int*>, "");
static_assert(!cuda::std::indirectly_movable<int[2], int[2]>, "");
static_assert(!cuda::std::indirectly_movable<int (&)[2], int (&)[2]>, "");

// Can't move between non-pointer types.
static_assert(!cuda::std::indirectly_movable<int*, int>, "");
static_assert(!cuda::std::indirectly_movable<int, int*>, "");
static_assert(!cuda::std::indirectly_movable<int, int>, "");

// Check some less common types.
static_assert(!cuda::std::indirectly_movable<void*, void*>, "");
static_assert(!cuda::std::indirectly_movable<int*, void*>, "");
static_assert(!cuda::std::indirectly_movable<int(), int()>, "");
static_assert(!cuda::std::indirectly_movable<int*, int()>, "");
static_assert(!cuda::std::indirectly_movable<void, void>, "");

// Can move move-only objects.
static_assert(cuda::std::indirectly_movable<MoveOnly*, MoveOnly*>, "");
static_assert(!cuda::std::indirectly_movable<MoveOnly*, const MoveOnly*>, "");
static_assert(!cuda::std::indirectly_movable<const MoveOnly*, const MoveOnly*>, "");
static_assert(!cuda::std::indirectly_movable<const MoveOnly*, MoveOnly*>, "");

template <class T>
struct PointerTo
{
  using value_type = T;
  __host__ __device__ T& operator*() const;
};

// Can copy through a dereferenceable class.
static_assert(cuda::std::indirectly_movable<int*, PointerTo<int>>, "");
static_assert(!cuda::std::indirectly_movable<int*, PointerTo<const int>>, "");
static_assert(cuda::std::indirectly_copyable<PointerTo<int>, PointerTo<int>>, "");
static_assert(!cuda::std::indirectly_copyable<PointerTo<int>, PointerTo<const int>>, "");
static_assert(cuda::std::indirectly_movable<MoveOnly*, PointerTo<MoveOnly>>, "");
static_assert(cuda::std::indirectly_movable<PointerTo<MoveOnly>, MoveOnly*>, "");
static_assert(cuda::std::indirectly_movable<PointerTo<MoveOnly>, PointerTo<MoveOnly>>, "");

int main(int, char**)
{
  return 0;
}
