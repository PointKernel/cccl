//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Temporary nvcc workaround for a cuda::buffer destructor conflict.
#if defined(__CUDACC__)
#  pragma nv_diag_suppress 20011
#endif

// Include the ref first to exercise it independently of the owning container's headers.
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map_ref.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

template <int N>
using int_c = ::cuda::std::integral_constant<int, N>;

using cg_sizes       = c2h::type_list<int_c<1>, int_c<4>>;
using capacity_kinds = c2h::type_list<::cuda::std::false_type, ::cuda::std::true_type>;

template <class Pair>
struct iota_pair
{
  int payload_offset;

  __host__ __device__ Pair operator()(int key) const noexcept
  {
    return Pair{key, key + payload_offset};
  }
};

struct is_even
{
  __device__ bool operator()(int value) const noexcept
  {
    return value % 2 == 0;
  }
};

template <class Value>
struct record_visit
{
  int* visits;
  int payload_offset;

  __device__ void operator()(Value slot) const noexcept
  {
    if (slot.second == slot.first + payload_offset)
    {
      ::cuda::atomic_ref<int, ::cuda::thread_scope_device>{visits[slot.first]}.fetch_add(
        1, ::cuda::memory_order_relaxed);
    }
  }
};

template <class Buffer>
c2h::host_vector<int> copy_to_host(::cuda::stream_ref stream, const Buffer& values)
{
  c2h::host_vector<int> result(values.size());
  REQUIRE(
    cudaMemcpyAsync(result.data(), values.data(), values.size() * sizeof(int), cudaMemcpyDeviceToHost, stream.get())
    == cudaSuccess);
  stream.sync();
  return result;
}

template <class Buffer, class Expected>
void require_values(::cuda::stream_ref stream, const Buffer& values, Expected expected)
{
  const auto actual = copy_to_host(stream, values);
  for (::cuda::std::size_t i = 0; i < actual.size(); ++i)
  {
    CAPTURE(i);
    REQUIRE(actual[i] == expected(static_cast<int>(i)));
  }
}

C2H_TEST("fixed_capacity_map_ref host operations over external storage", "[container][ref]", cg_sizes, capacity_kinds)
{
  constexpr int cg_size   = c2h::get<0, TestType>::value;
  constexpr bool fixed    = c2h::get<1, TestType>::value;
  constexpr int num_keys  = 64;
  constexpr int queries   = num_keys + 16;
  using probing_type      = cudax::cuco::linear_probing<cg_size, ::cuda::hash<int>>;
  constexpr auto capacity = cudax::cuco::make_valid_capacity<probing_type, 2>(::cuda::std::size_t{256});
  constexpr auto extent   = fixed ? capacity : ::cuda::std::dynamic_extent;
  using ref_type          = cudax::cuco::
    fixed_capacity_map_ref<int, int, ::cuda::thread_scope_device, ::cuda::std::equal_to<int>, probing_type, 2, extent>;
  using value_type = typename ref_type::value_type;
  static_assert(::cuda::std::is_trivially_copyable_v<ref_type>);

  const bool async = GENERATE(false, true);
  CAPTURE(cg_size, fixed, async);

  const ::cuda::stream stream{::cuda::device_ref{0}};
  const auto mr = ::cuda::device_default_memory_pool(stream.device());
  ::cuda::device_buffer<value_type> slots{stream, mr, capacity, ::cuda::no_init};
  const ref_type ref{
    cudax::cuco::empty_key{-1},
    cudax::cuco::empty_value{-1},
    ::cuda::std::equal_to<int>{},
    probing_type{},
    typename ref_type::storage_span_type{slots.data(), capacity}};
  const auto keys             = ::cuda::counting_iterator<int>{0};
  const auto pairs            = ::cuda::transform_iterator{keys, iota_pair<value_type>{7}};
  const auto reassigned_pairs = ::cuda::transform_iterator{keys, iota_pair<value_type>{99}};
  auto found                  = ::cuda::make_buffer<int>(stream, mr, queries, 42);
  auto present                = ::cuda::make_buffer<int>(stream, mr, queries, 42);

  ref.clear_async(stream);
  REQUIRE(ref.size(stream, mr) == 0);
  REQUIRE(ref.insert(stream, pairs, pairs + num_keys / 2, mr) == num_keys / 2);
  REQUIRE(ref.insert(stream, pairs, pairs + num_keys / 2, mr) == 0);

  // Offset the stencil so that testing input keys instead would select the wrong elements.
  REQUIRE(ref.insert_if(stream, pairs + num_keys / 2, pairs + num_keys, keys + num_keys / 2 + 1, is_even{}, mr)
          == num_keys / 4);
  REQUIRE(ref.size(stream, mr) == 3 * num_keys / 4);
  ref.insert_if_async(stream, pairs + num_keys / 2, pairs + num_keys, keys + num_keys / 2, is_even{});
  REQUIRE(ref.size(stream, mr) == num_keys);

  if (async)
  {
    ref.find_async(stream, keys, keys + queries, found.begin());
    ref.contains_async(stream, keys, keys + queries, present.begin());
  }
  else
  {
    ref.find(stream, keys, keys + queries, found.begin());
    ref.contains(stream, keys, keys + queries, present.begin());
  }
  require_values(stream, found, [](int i) {
    return i < num_keys ? i + 7 : -1;
  });
  require_values(stream, present, [](int i) {
    return i < num_keys;
  });

  if (async)
  {
    ref.find_if_async(stream, keys, keys + queries, keys + 1, is_even{}, found.begin());
    ref.contains_if_async(stream, keys, keys + queries, keys + 1, is_even{}, present.begin());
  }
  else
  {
    ref.find_if(stream, keys, keys + queries, keys + 1, is_even{}, found.begin());
    ref.contains_if(stream, keys, keys + queries, keys + 1, is_even{}, present.begin());
  }
  require_values(stream, found, [](int i) {
    return i < num_keys && i % 2 != 0 ? i + 7 : -1;
  });
  require_values(stream, present, [](int i) {
    return i < num_keys && i % 2 != 0;
  });

  auto inserted = ::cuda::make_buffer<int>(stream, mr, num_keys, 42);
  auto payloads = ::cuda::make_buffer<int>(stream, mr, num_keys, 42);
  if (async)
  {
    ref.insert_and_find_async(stream, reassigned_pairs, reassigned_pairs + num_keys, payloads.begin(), inserted.begin());
  }
  else
  {
    ref.insert_and_find(stream, reassigned_pairs, reassigned_pairs + num_keys, payloads.begin(), inserted.begin());
  }
  require_values(stream, payloads, [](int i) {
    return i + 7;
  });
  require_values(stream, inserted, [](int) {
    return 0;
  });

  ref.clear(stream);
  if (async)
  {
    ref.insert_and_find_async(stream, pairs, pairs + num_keys, payloads.begin(), inserted.begin());
    ref.insert_or_assign_async(stream, reassigned_pairs, reassigned_pairs + num_keys);
  }
  else
  {
    ref.insert_and_find(stream, pairs, pairs + num_keys, payloads.begin(), inserted.begin());
    ref.insert_or_assign(stream, reassigned_pairs, reassigned_pairs + num_keys);
  }
  require_values(stream, payloads, [](int i) {
    return i + 7;
  });
  require_values(stream, inserted, [](int) {
    return 1;
  });

  auto retrieved_keys               = ::cuda::make_buffer<int>(stream, mr, num_keys, -1);
  const auto [keys_end, values_end] = ref.retrieve_all(stream, retrieved_keys.begin(), payloads.begin(), mr);
  REQUIRE(keys_end == retrieved_keys.end());
  REQUIRE(values_end == payloads.end());
  const auto host_keys   = copy_to_host(stream, retrieved_keys);
  const auto host_values = copy_to_host(stream, payloads);
  c2h::host_vector<int> seen(num_keys, 0);
  for (int i = 0; i < num_keys; ++i)
  {
    REQUIRE(host_keys[i] >= 0);
    REQUIRE(host_keys[i] < num_keys);
    REQUIRE(seen[host_keys[i]]++ == 0);
    REQUIRE(host_values[i] == host_keys[i] + 99);
  }

  auto visits = ::cuda::make_buffer<int>(stream, mr, queries, 0);
  ref.for_each_async(stream, keys, keys + queries, record_visit<value_type>{visits.data(), 99});
  ref.for_each(stream, keys, keys + queries, record_visit<value_type>{visits.data(), 99});
  require_values(stream, visits, [](int i) {
    return i < num_keys ? 2 : 0;
  });

  ref.clear_async(stream);
  ref.insert_async(stream, pairs, pairs + num_keys);
  REQUIRE(ref.size(stream, mr) == num_keys);
  ref.clear(stream);
  REQUIRE(ref.size(stream, mr) == 0);

  // Empty ranges must preserve both storage and output values.
  REQUIRE(ref.insert(stream, pairs, pairs, mr) == 0);
  REQUIRE(ref.insert_if(stream, pairs, pairs, keys, is_even{}, mr) == 0);
  ref.insert_async(stream, pairs, pairs);
  ref.insert_if_async(stream, pairs, pairs, keys, is_even{});
  ref.insert_and_find(stream, pairs, pairs, payloads.begin(), inserted.begin());
  ref.insert_and_find_async(stream, pairs, pairs, payloads.begin(), inserted.begin());
  ref.insert_or_assign(stream, pairs, pairs);
  ref.insert_or_assign_async(stream, pairs, pairs);
  ref.find(stream, keys, keys, inserted.begin());
  ref.find_async(stream, keys, keys, inserted.begin());
  ref.find_if(stream, keys, keys, keys, is_even{}, inserted.begin());
  ref.find_if_async(stream, keys, keys, keys, is_even{}, inserted.begin());
  ref.contains(stream, keys, keys, inserted.begin());
  ref.contains_async(stream, keys, keys, inserted.begin());
  ref.contains_if(stream, keys, keys, keys, is_even{}, inserted.begin());
  ref.contains_if_async(stream, keys, keys, keys, is_even{}, inserted.begin());
  ref.for_each(stream, keys, keys, record_visit<value_type>{visits.data(), 99});
  ref.for_each_async(stream, keys, keys, record_visit<value_type>{visits.data(), 99});
  const auto [empty_keys_end, empty_values_end] =
    ref.retrieve_all(stream, retrieved_keys.begin(), payloads.begin(), mr);
  REQUIRE(empty_keys_end == retrieved_keys.begin());
  REQUIRE(empty_values_end == payloads.begin());
  REQUIRE(ref.size(stream, mr) == 0);
  require_values(stream, inserted, [](int) {
    return 1;
  });
  require_values(stream, visits, [](int i) {
    return i < num_keys ? 2 : 0;
  });
}

C2H_TEST("fixed_capacity_map owner and ref share host operations after rehash", "[container][ref]", cg_sizes)
{
  constexpr int cg_size = c2h::get<0, TestType>::value;
  using probing_type    = cudax::cuco::double_hashing<cg_size, ::cuda::hash<int>>;
  using map_type        = cudax::cuco::fixed_capacity_map<
    int,
    int,
    ::cuda::std::dynamic_extent,
    ::cuda::thread_scope_device,
    ::cuda::std::equal_to<int>,
    probing_type>;
  using value_type       = typename map_type::value_type;
  constexpr int num_keys = 64;
  const ::cuda::stream stream{::cuda::device_ref{0}};
  const auto mr = ::cuda::device_default_memory_pool(stream.device());
  map_type map{stream, mr, ::cuda::std::size_t{256}, cudax::cuco::empty_key{-1}, cudax::cuco::empty_value{-1}};
  const auto keys  = ::cuda::counting_iterator<int>{0};
  const auto pairs = ::cuda::transform_iterator{keys, iota_pair<value_type>{7}};
  auto found       = ::cuda::make_buffer<int>(stream, mr, num_keys, -1);

  {
    const auto ref = map.ref();
    REQUIRE(ref.insert(stream, pairs, pairs + num_keys / 2, mr) == num_keys / 2);
    map.insert_async(stream, pairs + num_keys / 2, pairs + num_keys);
    ref.find(stream, keys, keys + num_keys, found.begin());
    require_values(stream, found, [](int i) {
      return i + 7;
    });
    REQUIRE(map.size(stream) == ref.size(stream, mr));
  }

  // Rehash replaces the allocation, so obtain a fresh ref before using the new storage.
  map.rehash(stream, ::cuda::std::size_t{512});
  const auto ref = map.ref();
  REQUIRE(ref.size(stream, mr) == num_keys);
  const auto reassigned_pairs = ::cuda::transform_iterator{keys, iota_pair<value_type>{99}};
  ref.insert_or_assign_async(stream, reassigned_pairs, reassigned_pairs + num_keys);
  map.find(stream, keys, keys + num_keys, found.begin());
  require_values(stream, found, [](int i) {
    return i + 99;
  });
  ref.clear_async(stream);
  REQUIRE(map.size(stream) == 0);
}
