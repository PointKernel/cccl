//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_BULK_OPERATIONS_CUH
#define _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_BULK_OPERATIONS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

#  include <cub/device/device_for.cuh>
#  include <cub/device/device_reduce.cuh>
#  include <cub/device/device_select.cuh>
#  include <cub/device/device_transform.cuh>

#  include <cuda/__algorithm/copy.h>
#  include <cuda/__container/buffer.h>
#  include <cuda/__driver/driver_api.h>
#  include <cuda/__hierarchy/level_dimensions.h>
#  include <cuda/__iterator/constant_iterator.h>
#  include <cuda/__iterator/counting_iterator.h>
#  include <cuda/__iterator/transform_iterator.h>
#  include <cuda/__launch/configuration.h>
#  include <cuda/__launch/launch.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__functional/identity.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/span>

#  include <cuda/experimental/__cuco/detail/open_addressing/functors.cuh>
#  include <cuda/experimental/__cuco/detail/open_addressing/kernels.cuh>
#  include <cuda/experimental/__cuco/detail/open_addressing/slot_storage_ref.cuh>
#  include <cuda/experimental/__cuco/detail/utility/cuda.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__open_addressing
{
//! @brief Allocates a zero-initialized counter using the caller's temporary resource.
template <class _Size, class _MemoryResource>
[[nodiscard]] _CCCL_HOST_API ::cuda::device_buffer<_Size>
__make_counter(::cuda::stream_ref __stream, _MemoryResource __mr)
{
  return ::cuda::device_buffer<_Size>{__stream, __mr, {_Size{0}}};
}

//! @brief Reads a device counter and synchronizes the stream.
template <class _Size>
[[nodiscard]] _CCCL_HOST_API _Size
__read_counter(const ::cuda::device_buffer<_Size>& __counter, ::cuda::stream_ref __stream)
{
  _Size __result;

#  if _CCCL_CTK_AT_LEAST(13, 0)
  ::cuda::copy_configuration __config{};
  __config.src_access_order = ::cuda::source_access_order::stream;

  const ::cuda::std::span<_Size> __result_span{&__result, 1};
  ::cuda::copy_bytes(__stream, __counter, __result_span, __config);
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  ::cuda::__driver::__memcpyAsync(&__result, __counter.data(), sizeof(_Size), __stream.get());
#  endif // _CCCL_CTK_BELOW(13, 0)
  __stream.sync();
  return __result;
}

//! @brief Asynchronously fills existing slot storage with its empty sentinel.
template <class _Value, ::cuda::std::size_t _Extent>
_CCCL_HOST_API void
__clear_async(::cuda::stream_ref __stream, ::cuda::std::span<_Value, _Extent> __slots, _Value __empty_slot)
{
  if (__slots.empty())
  {
    return;
  }
  _CCCL_TRY_RUNTIME_API(
    CUB_NS_QUALIFIER::DeviceTransform::Fill,
    "cuco: failed to clear slot storage",
    __slots.data(),
    static_cast<detail::__index_type>(__slots.size()),
    __empty_slot,
    __stream);
}

//! @brief Inserts keys in `[first, last)` whose stencil satisfies `pred`.
//!
//! @return Number of successful insertions
template <class _InputIt, class _StencilIt, class _Predicate, class _Ref, class _MemoryResource>
_CCCL_HOST_API typename _Ref::size_type __insert_if(
  ::cuda::stream_ref __stream,
  _InputIt __first,
  _InputIt __last,
  _StencilIt __stencil,
  _Predicate __pred,
  _Ref __container_ref,
  _MemoryResource __mr)
{
  const auto __num_keys = detail::__distance(__first, __last);
  if (__num_keys == 0)
  {
    return 0;
  }

  auto __counter = __make_counter<typename _Ref::size_type>(__stream, __mr);

  const auto __grid_size = detail::__grid_size(__num_keys, _Ref::cg_size);

  const auto __config = ::cuda::make_config(
    ::cuda::grid_dims(static_cast<unsigned>(__grid_size)), ::cuda::block_dims<detail::__default_block_size>());
  using __kernel_type =
    void (*)(_InputIt, detail::__index_type, _StencilIt, _Predicate, typename _Ref::size_type*, _Ref);
  const auto __kernel = static_cast<__kernel_type>(
    __open_addressing::
      __insert_if_n<_Ref::cg_size, detail::__default_block_size, _InputIt, _StencilIt, _Predicate, _Ref>);
  ::cuda::launch(
    __stream, __config, __kernel, __first, __num_keys, __stencil, __pred, __counter.data(), __container_ref);

  return __read_counter(__counter, __stream);
}

//! @brief Inserts keys in `[first, last)` and returns the number of successful insertions.
template <class _InputIt, class _Ref, class _MemoryResource>
_CCCL_HOST_API typename _Ref::size_type
__insert(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _Ref __container_ref, _MemoryResource __mr)
{
  return __insert_if(
    __stream, __first, __last, ::cuda::constant_iterator<bool>{true}, ::cuda::std::identity{}, __container_ref, __mr);
}

//! @brief Asynchronously inserts keys in `[first, last)` whose stencil satisfies `pred`.
//!
//! @throws cuda_error if the insert operation fails to launch
template <class _InputIt, class _StencilIt, class _Predicate, class _Ref>
_CCCL_HOST_API void __insert_if_async(
  ::cuda::stream_ref __stream,
  _InputIt __first,
  _InputIt __last,
  _StencilIt __stencil,
  _Predicate __pred,
  _Ref __container_ref)
{
  const auto __num_keys = detail::__distance(__first, __last);
  if (__num_keys == 0)
  {
    return;
  }

  if constexpr (_Ref::cg_size == 1)
  {
    __open_addressing::__insert_if_fn __op{__first, __stencil, __pred, __container_ref};
    _CCCL_TRY_RUNTIME_API(CUB_NS_QUALIFIER::DeviceFor::Bulk, "cuco: failed to insert keys", __num_keys, __op, __stream);
  }
  else
  {
    const auto __grid_size = detail::__grid_size(__num_keys, _Ref::cg_size);

    const auto __config = ::cuda::make_config(
      ::cuda::grid_dims(static_cast<unsigned>(__grid_size)), ::cuda::block_dims<detail::__default_block_size>());
    using __kernel_type = void (*)(_InputIt, detail::__index_type, _StencilIt, _Predicate, _Ref);
    const auto __kernel = static_cast<__kernel_type>(
      __open_addressing::
        __insert_if_n<_Ref::cg_size, detail::__default_block_size, _InputIt, _StencilIt, _Predicate, _Ref>);
    ::cuda::launch(__stream, __config, __kernel, __first, __num_keys, __stencil, __pred, __container_ref);
  }
}

//! @brief Asynchronously inserts keys in `[first, last)`.
//!
//! @throws cuda_error if the insert operation fails to launch
template <class _InputIt, class _Ref>
_CCCL_HOST_API void __insert_async(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _Ref __container_ref)
{
  __insert_if_async(
    __stream, __first, __last, ::cuda::constant_iterator<bool>{true}, ::cuda::std::identity{}, __container_ref);
}

//! @brief Asynchronously inserts each element and returns its mapped value and insertion status.
//!
//! @throws cuda_error if the insert operation fails to launch
//!
//! @tparam _InputIt Device accessible random access input iterator
//! @tparam _FoundIt Device accessible output iterator assignable from the mapped type
//! @tparam _InsertedIt Device accessible output iterator assignable from bool
//! @tparam _Ref Device reference to the map
//!
//! @param[in] __stream CUDA stream used for insertion
//! @param[in] __first Beginning of the input sequence
//! @param[in] __last End of the input sequence
//! @param[out] __found_begin Beginning of the mapped-value output sequence
//! @param[out] __inserted_begin Beginning of the insertion-status output sequence
//! @param[in,out] __container_ref Map in which to insert the input pairs
template <class _InputIt, class _FoundIt, class _InsertedIt, class _Ref>
_CCCL_HOST_API void __insert_and_find_async(
  ::cuda::stream_ref __stream,
  _InputIt __first,
  _InputIt __last,
  _FoundIt __found_begin,
  _InsertedIt __inserted_begin,
  _Ref __container_ref)
{
  const auto __num_keys = detail::__distance(__first, __last);
  if (__num_keys == 0)
  {
    return;
  }

  const auto __grid_size = detail::__grid_size(__num_keys, _Ref::cg_size);
  const auto __config    = ::cuda::make_config(
    ::cuda::block_dims<detail::__default_block_size>(), ::cuda::grid_dims(static_cast<unsigned>(__grid_size)));
  ::cuda::launch(
    __stream,
    __config,
    __open_addressing::
      __insert_and_find_n<_Ref::cg_size, detail::__default_block_size, _InputIt, _FoundIt, _InsertedIt, _Ref>,
    __first,
    __num_keys,
    __found_begin,
    __inserted_begin,
    __container_ref);
}

//! @brief Asynchronously inserts or assigns pairs in `[__first, __last)`.
//!
//! @throws cuda_error if the operation fails to launch
template <class _InputIt, class _Ref>
_CCCL_HOST_API void
__insert_or_assign_async(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _Ref __container_ref)
{
  const auto __num_keys = detail::__distance(__first, __last);
  if (__num_keys == 0)
  {
    return;
  }
  const auto __grid_size = detail::__grid_size(__num_keys, _Ref::cg_size);
  const auto __config    = ::cuda::make_config(
    ::cuda::grid_dims(static_cast<unsigned>(__grid_size)), ::cuda::block_dims<detail::__default_block_size>());
  const auto& __kernel =
    __open_addressing::__insert_or_assign_n<_Ref::cg_size, detail::__default_block_size, _InputIt, _Ref>;
  ::cuda::launch(__stream, __config, __kernel, __first, __num_keys, __container_ref);
}

//! @brief Asynchronously checks if keys in `[first, last)` whose stencil satisfies `pred` exist.
//!
//! For each key `first[i]`, writes whether the key is present when `pred(stencil[i])` is true;
//! otherwise writes false.
//!
//! @throws cuda_error if the query operation fails to launch
template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt, class _Ref>
_CCCL_HOST_API void __contains_if_async(
  ::cuda::stream_ref __stream,
  _InputIt __first,
  _InputIt __last,
  _StencilIt __stencil,
  _Predicate __pred,
  _OutputIt __output_begin,
  _Ref __container_ref)
{
  const auto __num_keys = detail::__distance(__first, __last);
  if (__num_keys == 0)
  {
    return;
  }

  if constexpr (_Ref::cg_size == 1)
  {
    __open_addressing::__contains_if_fn __op{__first, __stencil, __pred, __output_begin, __container_ref};
    _CCCL_TRY_RUNTIME_API(CUB_NS_QUALIFIER::DeviceFor::Bulk, "cuco: failed to query keys", __num_keys, __op, __stream);
  }
  else
  {
    const auto __grid_size = detail::__grid_size(__num_keys, _Ref::cg_size);

    const auto __config = ::cuda::make_config(
      ::cuda::grid_dims(static_cast<unsigned>(__grid_size)), ::cuda::block_dims<detail::__default_block_size>());
    const auto& __kernel = __open_addressing::
      __contains_if_n<_Ref::cg_size, detail::__default_block_size, _InputIt, _StencilIt, _Predicate, _OutputIt, _Ref>;
    ::cuda::launch(
      __stream, __config, __kernel, __first, __num_keys, __stencil, __pred, __output_begin, __container_ref);
  }
}

//! @brief Asynchronously checks if keys in `[first, last)` exist in the container.
//!
//! @throws cuda_error if the query operation fails to launch
template <class _InputIt, class _OutputIt, class _Ref>
_CCCL_HOST_API void __contains_async(
  ::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _OutputIt __output_begin, _Ref __container_ref)
{
  __contains_if_async(
    __stream,
    __first,
    __last,
    ::cuda::constant_iterator<bool>{true},
    ::cuda::std::identity{},
    __output_begin,
    __container_ref);
}

//! @brief Asynchronously finds payloads for keys in `[first, last)` whose stencil satisfies `pred`.
//!
//! For each key `first[i]` with `pred(stencil[i]) == true` that is present, the associated payload is
//! written to the corresponding output position; otherwise the empty value sentinel is written.
//!
//! @throws cuda_error if the query operation fails to launch
template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt, class _Ref>
_CCCL_HOST_API void __find_if_async(
  ::cuda::stream_ref __stream,
  _InputIt __first,
  _InputIt __last,
  _StencilIt __stencil,
  _Predicate __pred,
  _OutputIt __output_begin,
  _Ref __container_ref)
{
  const auto __num_keys = detail::__distance(__first, __last);
  if (__num_keys == 0)
  {
    return;
  }

  const auto __grid_size = detail::__grid_size(__num_keys, _Ref::cg_size);

  const auto __config = ::cuda::make_config(
    ::cuda::grid_dims(static_cast<unsigned>(__grid_size)), ::cuda::block_dims<detail::__default_block_size>());
  const auto& __kernel = __open_addressing::
    __find_if_n<_Ref::cg_size, detail::__default_block_size, _InputIt, _StencilIt, _Predicate, _OutputIt, _Ref>;
  ::cuda::launch(__stream, __config, __kernel, __first, __num_keys, __stencil, __pred, __output_begin, __container_ref);
}

//! @brief Asynchronously finds the payloads for keys in `[first, last)`.
//!
//! For each key that is present, the associated payload is written to the corresponding output
//! position; for each key that is absent, the empty value sentinel is written instead.
//!
//! @throws cuda_error if the query operation fails to launch
template <class _InputIt, class _OutputIt, class _Ref>
_CCCL_HOST_API void __find_async(
  ::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _OutputIt __output_begin, _Ref __container_ref)
{
  __find_if_async(
    __stream,
    __first,
    __last,
    ::cuda::constant_iterator<bool>{true},
    ::cuda::std::identity{},
    __output_begin,
    __container_ref);
}

//! @brief Asynchronously applies `__callback_op` to a copy of every slot matching each key in
//! `[__first, __last)`.
//!
//! @note The return value of `__callback_op`, if any, is ignored.
//!
//! @throws cuda_error if the query operation fails to launch
template <class _InputIt, class _CallbackOp, class _Ref>
_CCCL_HOST_API void __for_each_async(
  ::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _CallbackOp __callback_op, _Ref __container_ref)
{
  const auto __num_keys = detail::__distance(__first, __last);
  if (__num_keys == 0)
  {
    return;
  }

  if constexpr (_Ref::cg_size == 1)
  {
    __open_addressing::__for_each_fn __op{__first, __callback_op, __container_ref};
    _CCCL_TRY_RUNTIME_API(CUB_NS_QUALIFIER::DeviceFor::Bulk, "cuco: failed to query keys", __num_keys, __op, __stream);
  }
  else
  {
    const auto __grid_size = detail::__grid_size(__num_keys, _Ref::cg_size);
    const auto __config    = ::cuda::make_config(
      ::cuda::grid_dims(static_cast<unsigned>(__grid_size)), ::cuda::block_dims<detail::__default_block_size>());
    const auto& __kernel =
      __open_addressing::__for_each_n<_Ref::cg_size, detail::__default_block_size, _InputIt, _CallbackOp, _Ref>;
    ::cuda::launch(__stream, __config, __kernel, __first, __num_keys, __callback_op, __container_ref);
  }
}

//! @brief Retrieves occupied slots and returns the output end, synchronizing the stream.
template <class _OutputIt, class _Ref, class _MemoryResource>
[[nodiscard]] _CCCL_HOST_API _OutputIt
__retrieve_all(::cuda::stream_ref __stream, _OutputIt __output_begin, _Ref __container_ref, _MemoryResource __mr)
{
  using __size_type        = typename _Ref::size_type;
  using __storage_ref_type = __slot_storage_ref<typename _Ref::value_type, _Ref::bucket_size, _Ref::capacity_v>;
  const auto __slots       = __container_ref.storage_span();
  const auto __storage     = __storage_ref_type{__slots.data(), __slots.size()};
  auto __counter           = __make_counter<__size_type>(__stream, __mr);

  const auto __input_begin = ::cuda::make_transform_iterator(
    ::cuda::counting_iterator<__size_type>{0}, __get_slot<true, __storage_ref_type>{__storage});
  const auto __is_filled = __slot_is_filled<true, typename _Ref::key_type>{
    __container_ref.empty_key_sentinel(), __container_ref.erased_key_sentinel()};
  const auto __env = ::cuda::std::execution::env{__stream, __mr};

  _CCCL_TRY_RUNTIME_API(
    CUB_NS_QUALIFIER::DeviceSelect::If,
    "cuco: failed to retrieve all elements",
    __input_begin,
    __output_begin,
    __counter.data(),
    __container_ref.capacity(),
    __is_filled,
    __env);

  return __output_begin + __read_counter(__counter, __stream);
}

//! @brief Counts occupied slots and synchronizes the stream.
template <class _Ref, class _MemoryResource>
[[nodiscard]] _CCCL_HOST_API typename _Ref::size_type
__size(::cuda::stream_ref __stream, _Ref __container_ref, _MemoryResource __mr)
{
  using __size_type        = typename _Ref::size_type;
  using __storage_ref_type = __slot_storage_ref<typename _Ref::value_type, _Ref::bucket_size, _Ref::capacity_v>;
  const auto __slots       = __container_ref.storage_span();
  const auto __storage     = __storage_ref_type{__slots.data(), __slots.size()};
  auto __counter           = __make_counter<__size_type>(__stream, __mr);

  const auto __input_begin = ::cuda::make_transform_iterator(
    ::cuda::counting_iterator<__size_type>{0}, __get_slot<true, __storage_ref_type>{__storage});
  const auto __is_filled = __slot_is_filled<true, typename _Ref::key_type>{
    __container_ref.empty_key_sentinel(), __container_ref.erased_key_sentinel()};
  const auto __env = ::cuda::std::execution::env{__stream, __mr};

  _CCCL_TRY_RUNTIME_API(
    CUB_NS_QUALIFIER::DeviceReduce::TransformReduce,
    "cuco: failed to get the number of elements",
    __input_begin,
    __counter.data(),
    __container_ref.capacity(),
    ::cuda::std::plus<__size_type>{},
    __is_filled,
    __size_type{0},
    __env);

  return __read_counter(__counter, __stream);
}
} // namespace cuda::experimental::cuco::__open_addressing

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_BULK_OPERATIONS_CUH
