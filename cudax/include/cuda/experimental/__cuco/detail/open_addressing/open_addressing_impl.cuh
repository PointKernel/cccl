//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_IMPL_CUH
#define _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_IMPL_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_transform.cuh>

#include <cuda/__container/buffer.h>
#include <cuda/__hierarchy/level_dimensions.h>
#include <cuda/__launch/configuration.h>
#include <cuda/__launch/launch.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/__type_traits/is_bitwise_comparable.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/span>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/bulk_operations.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/functors.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/kernels.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/slot_storage_ref.cuh>
#include <cuda/experimental/__cuco/detail/utility/cuda.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !_CCCL_COMPILER(NVRTC)

namespace cuda::experimental::cuco::__open_addressing
{
//! @brief Open addressing implementation class.
//!
//! @note This class should NOT be used directly.
//!
//! @throw If the size of the given key type is larger than 8 bytes
//! @throw If the size of the given slot type is larger than 16 bytes
//! @throw If the given key type doesn't have unique object representations, i.e.,
//! `cuda::is_bitwise_comparable_v<_Key> == false`
//! @throw If the probing scheme type is not inherited from
//! `cuda::experimental::cuco::detail::__probing_scheme_base`
//!
//! @tparam _Key Type used for keys. Requires `cuda::is_bitwise_comparable_v<_Key>`
//! @tparam _Value Type used for storage values
//! @tparam _Scope The scope in which operations will be performed by individual threads
//! @tparam _KeyEqual Binary callable type used to compare two keys for equality
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _MemoryResource Type of memory resource used for device storage
template <class _Key,
          class _Value,
          ::cuda::thread_scope _Scope,
          class _KeyEqual,
          class _ProbingScheme,
          int _BucketSize,
          class _MemoryResource>
class __open_addressing_impl
{
public:
  using __key_type            = _Key;
  using __value_type          = _Value;
  using __probing_scheme_type = _ProbingScheme;
  using __hasher              = typename __probing_scheme_type::hasher;
  using __size_type           = ::cuda::std::size_t;
  using __key_equal           = _KeyEqual;
  using __storage_ref_type    = __slot_storage_ref<__value_type, _BucketSize>;

  static constexpr auto __has_payload  = !::cuda::std::is_same_v<_Key, _Value>;
  static constexpr auto __cg_size      = _ProbingScheme::cg_size;
  static constexpr auto __bucket_size  = _BucketSize;
  static constexpr auto __thread_scope = _Scope;

  static_assert(sizeof(_Key) <= 8, "Container does not support key types larger than 8 bytes.");
  static_assert(sizeof(_Value) <= 16, "Container does not support slot types larger than 16 bytes.");
  static_assert(::cuda::is_bitwise_comparable_v<_Key>,
                "Key type must have unique object representations or have been explicitly declared as safe for "
                "bitwise comparison via specialization of cuda::is_bitwise_comparable_v<Key>.");
  static_assert(::cuda::std::is_base_of_v<detail::__probing_scheme_base<_ProbingScheme::cg_size>, _ProbingScheme>,
                "ProbingScheme must inherit from cuda::experimental::cuco::detail::__probing_scheme_base");

private:
  __value_type __empty_slot_sentinel;
  __key_type __erased_key_sentinel;
  __key_equal __predicate;
  __probing_scheme_type __probing_scheme;
  mutable _MemoryResource __memory_resource;
  ::cuda::device_buffer<__value_type> __slots;

  //! @brief Computes the number of buckets for a requested capacity.
  [[nodiscard]] _CCCL_HOST_API static __size_type __compute_num_buckets(__size_type __requested_capacity)
  {
    return make_valid_capacity<_ProbingScheme, _BucketSize>(__requested_capacity) / _BucketSize;
  }

  //! @brief Computes the number of buckets for a given number of keys and load factor.
  [[nodiscard]] _CCCL_HOST_API static __size_type __compute_num_buckets(__size_type __n, double __load_factor)
  {
    return make_valid_capacity<_ProbingScheme, _BucketSize>(__n, __load_factor) / _BucketSize;
  }

  //! @brief Extracts the key from a slot.
  [[nodiscard]] _CCCL_HOST_API constexpr const __key_type& __extract_key(const __value_type& __slot) const noexcept
  {
    if constexpr (__has_payload)
    {
      return __slot.first;
    }
    else
    {
      return __slot;
    }
  }

public:
  //! @brief Constructs an open addressing implementation with the given capacity.
  _CCCL_HOST_API __open_addressing_impl(
    ::cuda::stream_ref __stream,
    _MemoryResource __mr,
    __size_type __capacity,
    __value_type __empty_slot_sentinel,
    const _KeyEqual& __pred,
    const _ProbingScheme& __probing_scheme)
      : __empty_slot_sentinel{__empty_slot_sentinel}
      , __erased_key_sentinel{__extract_key(__empty_slot_sentinel)}
      , __predicate{__pred}
      , __probing_scheme{__probing_scheme}
      , __memory_resource{__mr}
      , __slots{__stream, __mr, __compute_num_buckets(__capacity) * _BucketSize, ::cuda::no_init}
  {
    clear_async(__stream);
  }

  //! @brief Constructs an open addressing implementation with capacity derived from desired load
  //! factor.
  _CCCL_HOST_API __open_addressing_impl(
    ::cuda::stream_ref __stream,
    _MemoryResource __mr,
    __size_type __n,
    double __desired_load_factor,
    __value_type __empty_slot_sentinel,
    const _KeyEqual& __pred,
    const _ProbingScheme& __probing_scheme)
      : __empty_slot_sentinel{__empty_slot_sentinel}
      , __erased_key_sentinel{__extract_key(__empty_slot_sentinel)}
      , __predicate{__pred}
      , __probing_scheme{__probing_scheme}
      , __memory_resource{__mr}
      , __slots{__stream, __mr, __compute_num_buckets(__n, __desired_load_factor) * _BucketSize, ::cuda::no_init}
  {
    clear_async(__stream);
  }

  //! @brief Constructs an open addressing implementation with erasure support.
  _CCCL_HOST_API __open_addressing_impl(
    ::cuda::stream_ref __stream,
    _MemoryResource __mr,
    __size_type __capacity,
    __value_type __empty_slot_sentinel,
    __key_type __erased_key_sentinel,
    const _KeyEqual& __pred,
    const _ProbingScheme& __probing_scheme)
      : __empty_slot_sentinel{__empty_slot_sentinel}
      , __erased_key_sentinel{__erased_key_sentinel}
      , __predicate{__pred}
      , __probing_scheme{__probing_scheme}
      , __memory_resource{__mr}
      , __slots{__stream, __mr, __compute_num_buckets(__capacity) * _BucketSize, ::cuda::no_init}
  {
    if (empty_key_sentinel() == erased_key_sentinel())
    {
      _CCCL_THROW(::std::invalid_argument, "The empty key sentinel and erased key sentinel cannot be the same value.");
    }
    clear_async(__stream);
  }

  //! @brief Copy-constructs an open addressing implementation.
  _CCCL_HIDE_FROM_ABI __open_addressing_impl(const __open_addressing_impl&) = default;

  //! @brief Move-constructs an open addressing implementation.
  _CCCL_HIDE_FROM_ABI __open_addressing_impl(__open_addressing_impl&&) = default;

  __open_addressing_impl& operator=(const __open_addressing_impl&) = delete;

  //! @brief Move-assigns an open addressing implementation.
  _CCCL_HIDE_FROM_ABI __open_addressing_impl& operator=(__open_addressing_impl&&) = default;

  // NVCC requires a non-defaulted destructor to honor the host annotation. The slot buffer must
  // be destroyed on the host. Explicitly default the other special members to preserve their behavior.
  _CCCL_HOST_API ~__open_addressing_impl() {} // NOLINT(modernize-use-equals-default)

  //! @brief Asynchronously initializes the owned slots with the empty sentinel.
  _CCCL_HOST_API void clear_async(::cuda::stream_ref __stream)
  {
    __open_addressing::__clear_async(__stream, ::cuda::std::span{__slots.data(), __slots.size()}, __empty_slot_sentinel);
  }

  //! @brief Returns the resource used for temporary storage by host operations.
  [[nodiscard]] _CCCL_HOST_API _MemoryResource memory_resource() const
  {
    return __memory_resource;
  }

  //! @brief Asynchronously regenerates the container without changing its capacity.
  //!
  //! @tparam _Container Owning container type
  //!
  //! @param[in] __stream CUDA stream used for this operation
  //! @param[in] __container Owning container whose reference is rebuilt after storage replacement
  template <class _Container>
  _CCCL_HOST_API void rehash_async(::cuda::stream_ref __stream, const _Container& __container)
  {
    rehash_async(__stream, capacity(), __container);
  }

  //! @brief Asynchronously replaces the slot storage and reinserts all filled slots.
  //!
  //! @tparam _Container Owning container type
  //!
  //! @param[in] __stream CUDA stream used for this operation
  //! @param[in] __capacity Requested new capacity
  //! @param[in] __container Owning container whose reference is rebuilt after storage replacement
  template <class _Container>
  _CCCL_HOST_API void rehash_async(::cuda::stream_ref __stream, __size_type __capacity, const _Container& __container)
  {
    const auto __new_capacity = __compute_num_buckets(__capacity) * _BucketSize;
    ::cuda::device_buffer<__value_type> __new_slots{__stream, __memory_resource, __new_capacity, ::cuda::no_init};

    _CCCL_TRY_RUNTIME_API(
      CUB_NS_QUALIFIER::DeviceTransform::Fill,
      "cuco: failed to initialize rehashed slot storage",
      __new_slots.data(),
      static_cast<detail::__index_type>(__new_capacity),
      __empty_slot_sentinel,
      __stream);

    __slots.swap(__new_slots);

    if (!__new_slots.empty())
    {
      constexpr auto __block_size = detail::__default_block_size;
      const auto __grid_size      = detail::__grid_size(static_cast<detail::__index_type>(__new_slots.size()));
      const auto __old_storage = __storage_ref_type{__new_slots.data(), static_cast<__size_type>(__new_slots.size())};
      const auto __new_ref     = __container.ref();
      const auto __is_filled = __slot_is_filled<__has_payload, __key_type>{empty_key_sentinel(), erased_key_sentinel()};

      using __new_ref_type   = decltype(__container.ref());
      using __predicate_type = __slot_is_filled<__has_payload, __key_type>;
      const auto __config =
        ::cuda::make_config(::cuda::grid_dims(static_cast<unsigned>(__grid_size)), ::cuda::block_dims<__block_size>());
      const auto& __kernel =
        __open_addressing::__rehash<__block_size, __storage_ref_type, __new_ref_type, __predicate_type>;
      ::cuda::launch(__stream, __config, __kernel, __old_storage, __new_ref, __is_filled);
    }

    __new_slots.destroy(__stream);
  }

  //! @brief Returns the total number of slots.
  [[nodiscard]] _CCCL_HOST_API constexpr __size_type capacity() const noexcept
  {
    return static_cast<__size_type>(__slots.size());
  }

  //! @brief Returns a pointer to the underlying slot array.
  [[nodiscard]] _CCCL_HOST_API __value_type* data() const noexcept
  {
    return const_cast<__value_type*>(__slots.data());
  }

  //! @brief Returns the empty key sentinel.
  [[nodiscard]] _CCCL_HOST_API constexpr __key_type empty_key_sentinel() const noexcept
  {
    return __extract_key(__empty_slot_sentinel);
  }

  //! @brief Returns the erased key sentinel.
  [[nodiscard]] _CCCL_HOST_API constexpr __key_type erased_key_sentinel() const noexcept
  {
    return __erased_key_sentinel;
  }

  //! @brief Returns the key comparison function.
  [[nodiscard]] _CCCL_HOST_API constexpr __key_equal key_eq() const noexcept
  {
    return __predicate;
  }

  //! @brief Returns the probing scheme.
  [[nodiscard]] _CCCL_HOST_API constexpr __probing_scheme_type probing_scheme() const noexcept
  {
    return __probing_scheme;
  }

  //! @brief Returns the hash function.
  [[nodiscard]] _CCCL_HOST_API constexpr __hasher hash_function() const noexcept
  {
    return probing_scheme().hash_function();
  }

  //! @brief Returns a non-owning reference to the stored slots.
  [[nodiscard]] _CCCL_HOST_API __storage_ref_type storage_ref() const noexcept
  {
    return __storage_ref_type{const_cast<__value_type*>(__slots.data()), capacity()};
  }
};
} // namespace cuda::experimental::cuco::__open_addressing

#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_IMPL_CUH
