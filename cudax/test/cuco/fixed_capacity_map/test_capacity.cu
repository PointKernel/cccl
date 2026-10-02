//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

constexpr int empty_key   = -1;
constexpr int empty_value = -1;

C2H_TEST("fixed_capacity_map dynamic capacity — capacity() reflects the valid capacity", "[capacity][dynamic]")
{
  constexpr ::cuda::std::size_t requested = 1000;
  using dyn_map_t                         = cudax::cuco::fixed_capacity_map<int, int>;

  static_assert(!cuda::std::is_copy_constructible_v<dyn_map_t>);
  static_assert(!cuda::std::is_copy_assignable_v<dyn_map_t>);
  static_assert(cuda::std::is_nothrow_move_constructible_v<dyn_map_t>);
  static_assert(cuda::std::is_nothrow_move_assignable_v<dyn_map_t>);

  static_assert(dyn_map_t::capacity_v == ::cuda::std::dynamic_extent,
                "capacity_v must be dynamic_extent for dynamic-capacity maps");
  static_assert(dyn_map_t::ref_type::capacity_v == ::cuda::std::dynamic_extent,
                "ref capacity_v must be dynamic_extent for dynamic maps");

  const auto valid =
    cudax::cuco::make_valid_capacity<dyn_map_t::probing_scheme_type, dyn_map_t::bucket_size>(requested);

  const ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  const dyn_map_t map{stream, mr, requested, cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};
  REQUIRE(map.capacity() == valid);
  REQUIRE(map.capacity() >= requested);
}

C2H_TEST("fixed_capacity_map static capacity — valid capacity and capacity_v", "[capacity][static]")
{
  // Double hashing rounds a requested slot count up to a prime-cycle capacity, so the valid capacity
  // must be computed from the probing scheme and bucket size before it can name a static map type.
  using probing                         = cudax::cuco::double_hashing<1, cuda::hash<int>>;
  [[maybe_unused]] constexpr int bucket = 1;

  constexpr ::cuda::std::size_t requested = 1000;
  constexpr auto valid                    = cudax::cuco::make_valid_capacity<probing, bucket>(requested);
  static_assert(valid > requested, "1000 is not a valid double-hashing capacity; it rounds up");

  using smap_t =
    cudax::cuco::fixed_capacity_map<int, int, valid, ::cuda::thread_scope_device, ::cuda::std::equal_to<int>, probing, 1>;
  static_assert(smap_t::capacity_v == valid, "the map type carries the valid capacity, not the request");
  static_assert(smap_t::ref_type::capacity_v == valid, "the ref carries the same valid capacity");

  const ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  const smap_t map{stream, mr, cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};
  REQUIRE(map.capacity() == valid);
}

C2H_TEST("fixed_capacity_map dynamic extent — load factor constructor", "[capacity][dynamic][load_factor]")
{
  constexpr int num_elements   = 500;
  constexpr double load_factor = 0.5;

  const ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  const cudax::cuco::fixed_capacity_map<int, int> map{
    stream,
    mr,
    static_cast<::cuda::std::size_t>(num_elements),
    load_factor,
    cudax::cuco::empty_key{empty_key},
    cudax::cuco::empty_value{empty_value}};

  // With load_factor = 0.5 and 500 elements, capacity should be >= 1000
  REQUIRE(map.capacity() >= static_cast<::cuda::std::size_t>(num_elements / load_factor));
}

struct custom_key
{
  int value;
};

// This key deliberately has no operator==; equality is supplied by custom_key_equal.
CUDAX_CUCO_DECLARE_BITWISE_COMPARABLE(custom_key);

struct custom_key_equal
{
  __host__ __device__ bool operator()(custom_key lhs, custom_key rhs) const noexcept
  {
    return lhs.value == rhs.value;
  }
};

struct custom_key_hash
{
  __host__ __device__ auto operator()(custom_key key) const noexcept
  {
    return ::cuda::hash<int>{}(key.value);
  }
};

C2H_TEST("fixed_capacity_map no-erasure constructors accept keys without operator==", "[capacity][constructor]")
{
  using probing_type = cudax::cuco::linear_probing<1, custom_key_hash>;
  using dynamic_map  = cudax::cuco::fixed_capacity_map<
    custom_key,
    int,
    ::cuda::std::dynamic_extent,
    ::cuda::thread_scope_device,
    custom_key_equal,
    probing_type>;
  constexpr auto capacity = cudax::cuco::make_valid_capacity<probing_type, 1>(::cuda::std::size_t{128});
  using static_map        = cudax::cuco::
    fixed_capacity_map<custom_key, int, capacity, ::cuda::thread_scope_device, custom_key_equal, probing_type>;
  const ::cuda::stream stream{::cuda::device_ref{0}};
  const auto mr             = ::cuda::device_default_memory_pool(stream.device());
  const auto key_sentinel   = cudax::cuco::empty_key{custom_key{-1}};
  const auto value_sentinel = cudax::cuco::empty_value{-1};

  SECTION("dynamic capacity")
  {
    const dynamic_map map{stream, mr, ::cuda::std::size_t{128}, key_sentinel, value_sentinel};
    REQUIRE(map.size(stream) == 0);
  }
  SECTION("load factor")
  {
    const dynamic_map map{stream, mr, ::cuda::std::size_t{64}, 0.5, key_sentinel, value_sentinel};
    REQUIRE(map.size(stream) == 0);
  }
  SECTION("static capacity")
  {
    const static_map map{stream, mr, key_sentinel, value_sentinel};
    REQUIRE(map.size(stream) == 0);
  }
}
