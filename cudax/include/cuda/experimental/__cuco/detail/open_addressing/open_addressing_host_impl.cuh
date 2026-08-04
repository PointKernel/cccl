//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_HOST_IMPL_CUH
#define _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_HOST_IMPL_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_for.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/__container/buffer.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/__iterator/constant_iterator.h>
#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__iterator/transform_iterator.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/__cuco/detail/open_addressing/functors.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/kernels.cuh>
#include <cuda/experimental/__cuco/detail/utility/cuda.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !_CCCL_COMPILER(NVRTC)

namespace cuda::experimental::cuco::__open_addressing
{
template <class _Ref>
class __open_addressing_host_impl
{
  using __key_type   = typename _Ref::key_type;
  using __value_type = typename _Ref::value_type;
  using __size_type  = typename _Ref::size_type;

  static constexpr auto __has_payload = !::cuda::std::is_same_v<__key_type, __value_type>;
  static constexpr auto __cg_size     = _Ref::cg_size;

  template <class _MemoryResource>
  [[nodiscard]] _CCCL_HOST_API static ::cuda::device_buffer<__size_type>
  __make_counter(::cuda::stream_ref __stream, _MemoryResource& __memory_resource)
  {
    return ::cuda::device_buffer<__size_type>{__stream, __memory_resource, {__size_type{0}}};
  }

  [[nodiscard]] _CCCL_HOST_API static __size_type
  __read_counter(const ::cuda::device_buffer<__size_type>& __counter, ::cuda::stream_ref __stream)
  {
    __size_type __result;
    ::cuda::__driver::__memcpyAsync(&__result, __counter.data(), sizeof(__size_type), __stream.get());
    __stream.sync();
    return __result;
  }

public:
  _CCCL_HOST_API static void clear(::cuda::stream_ref __stream, _Ref __container_ref)
  {
    clear_async(__stream, __container_ref);
    __stream.sync();
  }

  _CCCL_HOST_API static void clear_async(::cuda::stream_ref __stream, _Ref __container_ref)
  {
    const auto __n = __container_ref.capacity();
    if (__n == 0)
    {
      return;
    }
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceTransform::Fill,
      "cuco: failed to clear slot storage",
      __container_ref.storage_span().data(),
      static_cast<detail::__index_type>(__n),
      __value_type{__container_ref.empty_key_sentinel(), __container_ref.empty_value_sentinel()},
      __stream);
  }

  template <class _MemoryResource, class _InputIt>
  [[nodiscard]] _CCCL_HOST_API static __size_type
  insert(::cuda::stream_ref __stream,
         _MemoryResource __memory_resource,
         _InputIt __first,
         _InputIt __last,
         _Ref __container_ref)
  {
    const auto __num_keys = detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return 0;
    }

    auto __counter         = __make_counter(__stream, __memory_resource);
    const auto __grid_size = detail::__grid_size(__num_keys, __cg_size);

    __open_addressing::__insert_if_n<__cg_size, detail::__default_block_size>
      <<<static_cast<unsigned>(__grid_size), detail::__default_block_size, 0, __stream.get()>>>(
        __first,
        __num_keys,
        ::cuda::constant_iterator<bool>{true},
        ::cuda::std::identity{},
        __counter.data(),
        __container_ref);

    return __read_counter(__counter, __stream);
  }

  template <class _InputIt>
  _CCCL_HOST_API static void
  insert_async(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _Ref __container_ref)
  {
    const auto __num_keys = detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    if constexpr (__cg_size == 1)
    {
      __open_addressing::__insert_if_fn __op{
        __first, ::cuda::constant_iterator<bool>{true}, ::cuda::std::identity{}, __container_ref};
      _CCCL_TRY_CUDA_API(CUB_NS_QUALIFIER::DeviceFor::Bulk, "cuco: failed to insert keys", __num_keys, __op, __stream);
    }
    else
    {
      const auto __grid_size = detail::__grid_size(__num_keys, __cg_size);
      __open_addressing::__insert_if_n<__cg_size, detail::__default_block_size>
        <<<static_cast<unsigned>(__grid_size), detail::__default_block_size, 0, __stream.get()>>>(
          __first, __num_keys, ::cuda::constant_iterator<bool>{true}, ::cuda::std::identity{}, __container_ref);
    }
  }

  template <class _InputIt, class _OutputIt>
  _CCCL_HOST_API static void contains_async(
    ::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _OutputIt __output_begin, _Ref __container_ref)
  {
    const auto __num_keys = detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    if constexpr (__cg_size == 1)
    {
      __open_addressing::__contains_if_fn __op{
        __first, ::cuda::constant_iterator<bool>{true}, ::cuda::std::identity{}, __output_begin, __container_ref};
      _CCCL_TRY_CUDA_API(CUB_NS_QUALIFIER::DeviceFor::Bulk, "cuco: failed to query keys", __num_keys, __op, __stream);
    }
    else
    {
      const auto __grid_size = detail::__grid_size(__num_keys, __cg_size);
      __open_addressing::__contains_if_n<__cg_size, detail::__default_block_size>
        <<<static_cast<unsigned>(__grid_size), detail::__default_block_size, 0, __stream.get()>>>(
          __first,
          __num_keys,
          ::cuda::constant_iterator<bool>{true},
          ::cuda::std::identity{},
          __output_begin,
          __container_ref);
    }
  }

  template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt>
  _CCCL_HOST_API static void find_if_async(
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

    const auto __grid_size = detail::__grid_size(__num_keys, __cg_size);
    __open_addressing::__find_if_n<__cg_size, detail::__default_block_size>
      <<<static_cast<unsigned>(__grid_size), detail::__default_block_size, 0, __stream.get()>>>(
        __first, __num_keys, __stencil, __pred, __output_begin, __container_ref);
  }

  template <class _InputIt, class _OutputIt>
  _CCCL_HOST_API static void find_async(
    ::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _OutputIt __output_begin, _Ref __container_ref)
  {
    find_if_async(
      __stream,
      __first,
      __last,
      ::cuda::constant_iterator<bool>{true},
      ::cuda::std::identity{},
      __output_begin,
      __container_ref);
  }

  template <class _MemoryResource, class _OutputIt>
  [[nodiscard]] _CCCL_HOST_API static _OutputIt retrieve_all(
    ::cuda::stream_ref __stream, _MemoryResource __memory_resource, _OutputIt __output_begin, _Ref __container_ref)
  {
    auto __counter           = __make_counter(__stream, __memory_resource);
    const auto __input_begin = ::cuda::make_transform_iterator(
      ::cuda::counting_iterator<__size_type>{0},
      __get_slot<__has_payload, typename _Ref::__storage_ref_type>{__container_ref.__impl.storage_ref()});
    const auto __is_filled = __slot_is_filled<__has_payload, __key_type>{
      __container_ref.empty_key_sentinel(), __container_ref.erased_key_sentinel()};
    const auto __env = ::cuda::std::execution::env{__stream, __memory_resource};

    _CCCL_TRY_CUDA_API(
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
};
} // namespace cuda::experimental::cuco::__open_addressing

#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_HOST_IMPL_CUH
