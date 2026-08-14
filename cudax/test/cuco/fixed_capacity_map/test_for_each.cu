//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Temporary nvcc workaround for a cuda::buffer destructor conflict
#if defined(__CUDACC__)
#  pragma nv_diag_suppress 20011
#endif // defined(__CUDACC__)

#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <cuda/__cccl_config>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <cooperative_groups.h>
#include <testing.cuh>

namespace cudax = cuda::experimental;
namespace cg    = cooperative_groups;

template <int N>
using int_c = ::cuda::std::integral_constant<int, N>;

using key_types     = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using cg_sizes      = c2h::type_list<int_c<1>, int_c<2>>;
using bucket_sizes  = c2h::type_list<int_c<1>, int_c<2>>;
using probing_kinds = c2h::type_list<int_c<0>, int_c<1>>; // 0 = linear probing, 1 = double hashing

constexpr int payload_offset = 7;

template <class Pair>
struct iota_pair
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API Pair operator()(typename Pair::first_type key) const noexcept
  {
    return Pair{key, static_cast<typename Pair::second_type>(key + payload_offset)};
  }
};

struct count_matching_even_keys
{
  int* count;

  template <class Pair>
  _CCCL_DEVICE_API void operator()(Pair slot) const noexcept
  {
    const auto expected_value = static_cast<typename Pair::second_type>(slot.first + payload_offset);
    if ((slot.first % 2 == 0) && (slot.second == expected_value))
    {
      ::cuda::atomic_ref<int, ::cuda::thread_scope_device>{*count}.fetch_add(1, ::cuda::std::memory_order_relaxed);
    }
  }
};

struct has_expected_count
{
  int expected;

  [[nodiscard]] _CCCL_DEVICE_API bool operator()(int count) const noexcept
  {
    return count == expected;
  }
};

template <class Ref, class InputIt, class CallbackOp>
__global__ void ref_for_each(Ref ref, InputIt first, int num_keys, CallbackOp callback)
{
  const auto block = cg::this_thread_block();
  const auto tile  = cg::tiled_partition<Ref::cg_size>(block);
  const auto index =
    (static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x)) / Ref::cg_size;

  if (index < num_keys)
  {
    const auto key = *(first + index);
    if constexpr (Ref::cg_size == 1)
    {
      ref.for_each(key, callback);
    }
    else
    {
      ref.for_each(tile, key, callback);
    }
  }
}

C2H_TEST("fixed_capacity_map for_each", "[container]", key_types, cg_sizes, bucket_sizes, probing_kinds)
{
  using key_type                             = c2h::get<0, TestType>;
  [[maybe_unused]] constexpr int cg_size     = c2h::get<1, TestType>::value;
  [[maybe_unused]] constexpr int bucket_size = c2h::get<2, TestType>::value;
  [[maybe_unused]] constexpr int probing     = c2h::get<3, TestType>::value;

  using hasher = ::cuda::hash<key_type>;
  using probing_type =
    ::cuda::std::conditional_t<probing == 0,
                               cudax::cuco::linear_probing<cg_size, hasher>,
                               cudax::cuco::double_hashing<cg_size, hasher>>;
  using map_type = cudax::cuco::fixed_capacity_map<
    key_type,
    key_type,
    ::cuda::std::dynamic_extent,
    ::cuda::thread_scope_device,
    ::cuda::std::equal_to<key_type>,
    probing_type,
    bucket_size>;
  using value_type = typename map_type::value_type;

  constexpr int num_keys        = 400;
  constexpr int num_counters    = 5;
  constexpr int threads         = 128;
  constexpr key_type empty_slot = key_type{-1};

  ::cuda::stream stream{::cuda::device_ref{0}};
  const auto mr = ::cuda::device_default_memory_pool(stream.device());

  map_type map{stream,
               mr,
               ::cuda::std::size_t{num_keys} * 2,
               cudax::cuco::empty_key{empty_slot},
               cudax::cuco::empty_value{empty_slot}};

  const auto pairs = ::cuda::transform_iterator(::cuda::counting_iterator<key_type>{0}, iota_pair<value_type>{});
  REQUIRE(map.insert(stream, pairs, pairs + num_keys) == num_keys);

  auto counters        = ::cuda::make_buffer<int>(stream, mr, num_counters, 0);
  const auto keys      = ::cuda::counting_iterator<key_type>{0};
  const auto keys_last = keys + 2 * num_keys;

  map.for_each(stream, count_matching_even_keys{counters.data()});
  map.for_each_async(stream, count_matching_even_keys{counters.data() + 1});
  map.for_each(stream, keys, keys_last, count_matching_even_keys{counters.data() + 2});
  map.for_each_async(stream, keys, keys_last, count_matching_even_keys{counters.data() + 3});

  constexpr int num_probe_keys = 2 * num_keys;
  constexpr int num_threads    = num_probe_keys * cg_size;
  constexpr int num_blocks     = (num_threads + threads - 1) / threads;
  ref_for_each<<<num_blocks, threads, 0, stream.get()>>>(
    map.ref(), keys, num_probe_keys, count_matching_even_keys{counters.data() + 4});
  REQUIRE(cudaGetLastError() == cudaSuccess);

  REQUIRE(::thrust::all_of(
    ::thrust::cuda::par.on(stream.get()), counters.begin(), counters.end(), has_expected_count{num_keys / 2}));
}
