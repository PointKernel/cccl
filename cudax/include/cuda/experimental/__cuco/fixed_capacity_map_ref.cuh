//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_FIXED_CAPACITY_MAP_REF_CUH
#define _CUDAX___CUCO_FIXED_CAPACITY_MAP_REF_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__atomic/atomic.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/__type_traits/is_bitwise_comparable.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/span>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/detail/bitwise_compare.cuh>
#include <cuda/experimental/__cuco/detail/equal_wrapper.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/open_addressing_ref_impl.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/slot_storage_ref.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>
#include <cuda/experimental/__cuco/types.cuh>

#include <cooperative_groups.h>

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
#  include <cuda/__iterator/zip_iterator.h>
#  include <cuda/__launch/configuration.h>
#  include <cuda/__launch/launch.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__functional/identity.h>
#  include <cuda/std/__functional/operations.h>

#  include <cuda/experimental/__cuco/detail/open_addressing/functors.cuh>
#  include <cuda/experimental/__cuco/detail/open_addressing/kernels.cuh>
#  include <cuda/experimental/__cuco/detail/utility/cuda.cuh>
#endif

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief Non-owning reference type for `fixed_capacity_map` with host bulk and device operations.
//!
//! This lightweight, trivially-copyable reference is intended to be passed by value to device code
//! for performing insert and lookup operations on the hash map. Host bulk operations act on the
//! same storage and do not take ownership of it. Operations requiring temporary device storage
//! accept a memory resource as their final argument.
//!
//! @pre Host bulk operations require storage accessible to all launched device threads, with a
//! thread scope suitable for those threads. Block shared-memory refs support device operations
//! only. Storage must remain valid until all operations using the ref have completed.
//!
//! @note Concurrent modify and lookup on the same map are not supported: lookups perform non-atomic
//! loads, so a lookup must not run concurrently with an insert (doing so is a data race).
//! @note cuCollections data structures always place the slot keys on the right-hand side when
//! invoking the key comparison predicate, i.e., `__pred(__query_key, __slot_key)`.
//! @note `_ProbingScheme::cg_size` indicates how many threads are used to handle one independent
//! device operation. `cg_size == 1` uses the scalar (or non-CG) code paths.
//! @note `_Capacity` is a span-style `size_t` non-type parameter encoding the *requested* slot
//! count. Pass `cuda::std::dynamic_extent` (the default) for runtime-sized maps; any concrete
//! value encodes the requested slot count at compile time. The actual slot count is the
//! prime/stride-adjusted value exposed as `capacity_v` and matches the owning map's
//! `fixed_capacity_map::capacity_v` for the same parameters.
//!
//! @tparam _Key Type used for keys
//! @tparam _Tp Type used for mapped values. `insert_and_find` requires `cuda::is_bitwise_comparable_v<_Tp>`;
//! use `CUDAX_CUCO_DECLARE_BITWISE_COMPARABLE` to explicitly opt in when safe.
//! @tparam _Scope The scope in which operations will be performed by individual threads
//! @tparam _KeyEqual Binary callable type used to compare two keys for equality
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _Capacity Requested slot count, or `cuda::std::dynamic_extent` for runtime sizing
template <class _Key,
          class _Tp,
          ::cuda::thread_scope _Scope,
          class _KeyEqual,
          class _ProbingScheme,
          int _BucketSize,
          ::cuda::std::size_t _Capacity = ::cuda::std::dynamic_extent>
class fixed_capacity_map_ref
{
  static_assert(sizeof(_Key) <= 8, "Container does not support key types larger than 8 bytes.");
  static_assert(::cuda::is_power_of_two(sizeof(_Key)), "key_type size must be a power of two");
  static_assert(sizeof(_Tp) <= 8, "sizeof(mapped_type) must be no larger than 8 bytes.");
  static_assert(::cuda::is_power_of_two(sizeof(::cuda::std::pair<_Key, _Tp>)),
                "value_type size must be a power of two");
  static_assert(::cuda::is_bitwise_comparable_v<_Key>,
                "Key type must have unique object representations or have been explicitly declared as safe for "
                "bitwise comparison via specialization of cuda::is_bitwise_comparable_v<Key>.");

  static constexpr bool __allows_duplicates = false;

  static_assert(_Capacity == ::cuda::std::dynamic_extent || is_valid_capacity<_ProbingScheme, _BucketSize>(_Capacity),
                "Capacity must be a valid open-addressing capacity; obtain it via cuco::make_valid_capacity");

public:
  using key_type            = _Key; ///< Key type
  using mapped_type         = _Tp; ///< Payload (mapped value) type
  using value_type          = ::cuda::std::pair<_Key, _Tp>; ///< Key-payload pair type
  using probing_scheme_type = _ProbingScheme; ///< Probing scheme type
  using hasher              = typename probing_scheme_type::hasher; ///< Hash function type
  using size_type           = ::cuda::std::size_t; ///< Size type
  using key_equal           = _KeyEqual; ///< Key equality comparator type
  using iterator            = value_type*; ///< Slot iterator
  using const_iterator      = const value_type*; ///< Const slot iterator

  static constexpr auto cg_size      = probing_scheme_type::cg_size; ///< Cooperative-group size for probing
  static constexpr auto bucket_size  = _BucketSize; ///< Number of slots per bucket
  static constexpr auto thread_scope = _Scope; ///< CUDA thread scope for atomic operations

  //! @brief Compile-time adjusted slot count; `cuda::std::dynamic_extent` when `_Capacity` is dynamic.
  static constexpr size_type capacity_v = _Capacity;

  //! @brief Slot-storage span type. For static `_Capacity`, the span carries the adjusted
  //! `capacity_v` extent at compile time; for dynamic `_Capacity`, the extent is dynamic.
  using storage_span_type = ::cuda::std::span<value_type, capacity_v>;

private:
  // Internal adapter to the open-addressing impl. The storage's `_Capacity` template arg receives
  // the (already valid) `capacity_v`, so when `_Capacity` is static the slot count travels through
  // the storage's extent at compile time and the probing iterator's modular reduction folds to a
  // constant.
  using __storage_ref_type = __open_addressing::__slot_storage_ref<value_type, _BucketSize, capacity_v>;

  //! @brief Returns the slot count of the given span, validating it for the dynamic case.
  //!
  //! @param __slots Span over the slot storage
  //!
  //! @return The total slot count
  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr size_type __checked_capacity(storage_span_type __slots) noexcept
  {
    if constexpr (_Capacity == ::cuda::std::dynamic_extent)
    {
      _CCCL_ASSERT((is_valid_capacity<_ProbingScheme, _BucketSize>(__slots.size())),
                   "storage size is not a valid capacity");
    }
    return __slots.size();
  }

  using __impl_type = __open_addressing::
    __open_addressing_ref_impl<_Key, _Scope, _KeyEqual, _ProbingScheme, __storage_ref_type, __allows_duplicates>;

  __impl_type __impl;

public:
  //! @brief Constructs a ref without erasure support.
  //!
  //! @param __empty_key_sentinel Sentinel indicating an empty key slot
  //! @param __empty_value_sentinel Sentinel indicating an empty payload
  //! @param __predicate Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __slots Span over the slot storage; must contain `capacity()` slots
  _CCCL_HOST_DEVICE_API explicit constexpr fixed_capacity_map_ref(
    empty_key<_Key> __empty_key_sentinel,
    empty_value<_Tp> __empty_value_sentinel,
    const _KeyEqual& __predicate,
    const _ProbingScheme& __probing_scheme,
    storage_span_type __slots) noexcept
      : __impl{value_type{key_type(__empty_key_sentinel), mapped_type(__empty_value_sentinel)},
               __predicate,
               __probing_scheme,
               __storage_ref_type{__slots.data(), __checked_capacity(__slots)}}
  {}

  //! @brief Constructs a ref with erasure support.
  //!
  //! @param __empty_key_sentinel Sentinel indicating an empty key slot
  //! @param __empty_value_sentinel Sentinel indicating an empty payload
  //! @param __erased_key_sentinel Sentinel indicating an erased key slot
  //! @param __predicate Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __slots Span over the slot storage; must contain `capacity()` slots
  _CCCL_HOST_DEVICE_API explicit constexpr fixed_capacity_map_ref(
    empty_key<_Key> __empty_key_sentinel,
    empty_value<_Tp> __empty_value_sentinel,
    erased_key<_Key> __erased_key_sentinel,
    const _KeyEqual& __predicate,
    const _ProbingScheme& __probing_scheme,
    storage_span_type __slots) noexcept
      : __impl{value_type{key_type(__empty_key_sentinel), mapped_type(__empty_value_sentinel)},
               key_type(__erased_key_sentinel),
               __predicate,
               __probing_scheme,
               __storage_ref_type{__slots.data(), __checked_capacity(__slots)}}
  {}

  // ===== Accessors =====

  //! @brief Returns the total number of slots.
  //!
  //! @return Total slot count (equal to the owning map's `capacity()`)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr size_type capacity() const noexcept
  {
    return __impl.capacity();
  }

  //! @brief Returns the sentinel value used to represent an empty key slot.
  //!
  //! @return The sentinel value used to represent an empty key slot
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr key_type empty_key_sentinel() const noexcept
  {
    return __impl.empty_key_sentinel();
  }

  //! @brief Returns the sentinel value used to represent an empty payload slot.
  //!
  //! @return The sentinel value used to represent an empty payload slot
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr mapped_type empty_value_sentinel() const noexcept
  {
    return __impl.empty_value_sentinel();
  }

  //! @brief Returns the sentinel value used to represent an erased key slot.
  //!
  //! @return The sentinel value used to represent an erased key slot
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr key_type erased_key_sentinel() const noexcept
  {
    return __impl.erased_key_sentinel();
  }

  //! @brief Returns the function used to compare keys for equality.
  //!
  //! @return The key equality comparator
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr key_equal key_eq() const noexcept
  {
    return __impl.key_eq();
  }

  //! @brief Returns the function(s) used to hash keys.
  //!
  //! @return The hasher used by this ref's probing scheme
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr hasher hash_function() const noexcept
  {
    return __impl.hash_function();
  }

  //! @brief Returns the probing scheme used to resolve hash collisions.
  //!
  //! @return The probing scheme object
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr probing_scheme_type probing_scheme() const noexcept
  {
    return __impl.probing_scheme();
  }

  // ===== Rebind =====

  //! @brief Makes a copy of this ref with the given key comparator.
  //!
  //! @tparam _NewKeyEqual New key comparator type
  //!
  //! @param __predicate New key comparator
  //!
  //! @return Copy of this ref using the new key comparator
  template <class _NewKeyEqual>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto rebind_key_eq(const _NewKeyEqual& __predicate) const noexcept
  {
    using __rebound_ref =
      fixed_capacity_map_ref<_Key, _Tp, _Scope, _NewKeyEqual, _ProbingScheme, _BucketSize, _Capacity>;

    return ::cuda::experimental::cuco::detail::__bitwise_compare(empty_key_sentinel(), erased_key_sentinel())
           ? __rebound_ref{empty_key<_Key>{empty_key_sentinel()},
                           empty_value<_Tp>{empty_value_sentinel()},
                           __predicate,
                           probing_scheme(),
                           storage_span()}
           : __rebound_ref{empty_key<_Key>{empty_key_sentinel()},
                           empty_value<_Tp>{empty_value_sentinel()},
                           erased_key<_Key>{erased_key_sentinel()},
                           __predicate,
                           probing_scheme(),
                           storage_span()};
  }

  //! @brief Makes a copy of this ref with the given hash function.
  //!
  //! @tparam _NewHash New hash function type
  //!
  //! @param __hash New hash function
  //!
  //! @return Copy of this ref using the new hash function
  template <class _NewHash>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto rebind_hash_function(const _NewHash& __hash) const
  {
    const auto __probing_scheme = probing_scheme().rebind_hash_function(__hash);
    using __rebound_ref =
      fixed_capacity_map_ref<_Key,
                             _Tp,
                             _Scope,
                             _KeyEqual,
                             ::cuda::std::decay_t<decltype(__probing_scheme)>,
                             _BucketSize,
                             _Capacity>;

    return ::cuda::experimental::cuco::detail::__bitwise_compare(empty_key_sentinel(), erased_key_sentinel())
           ? __rebound_ref{empty_key<_Key>{empty_key_sentinel()},
                           empty_value<_Tp>{empty_value_sentinel()},
                           key_eq(),
                           __probing_scheme,
                           storage_span()}
           : __rebound_ref{empty_key<_Key>{empty_key_sentinel()},
                           empty_value<_Tp>{empty_value_sentinel()},
                           erased_key<_Key>{erased_key_sentinel()},
                           key_eq(),
                           __probing_scheme,
                           storage_span()};
  }

  //! @brief Returns a const iterator to one past the last slot (the end sentinel).
  //!
  //! @return Past-the-end const iterator
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const_iterator end() const noexcept
  {
    return __impl.end();
  }

  //! @brief Returns an iterator to one past the last slot (the end sentinel).
  //!
  //! @return Past-the-end iterator
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr iterator end() noexcept
  {
    return __impl.end();
  }

  //! @brief Returns a span over the slot storage backing this ref.
  //!
  //! @return Span of `capacity()` slots
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr storage_span_type storage_span() const noexcept
  {
    return storage_span_type{__impl.storage_ref().data(), __impl.capacity()};
  }

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

private:
  [[nodiscard]] _CCCL_HOST_API static auto __launch_config(detail::__index_type __num_keys)
  {
    const auto __grid_size = detail::__grid_size(__num_keys, cg_size);
    return ::cuda::make_config(
      ::cuda::grid_dims(static_cast<unsigned>(__grid_size)), ::cuda::block_dims<detail::__default_block_size>());
  }

  template <class _MemoryResource>
  [[nodiscard]] _CCCL_HOST_API static ::cuda::device_buffer<size_type>
  __make_counter(::cuda::stream_ref __stream, _MemoryResource __mr)
  {
    return ::cuda::device_buffer<size_type>{__stream, __mr, {size_type{0}}};
  }

  [[nodiscard]] _CCCL_HOST_API static size_type
  __read_counter(const ::cuda::device_buffer<size_type>& __counter, ::cuda::stream_ref __stream)
  {
    size_type __result;
#  if _CCCL_CTK_AT_LEAST(13, 0)
    ::cuda::copy_configuration __config{};
    __config.src_access_order = ::cuda::source_access_order::stream;
    const ::cuda::std::span<size_type> __result_span{&__result, 1};
    ::cuda::copy_bytes(__stream, __counter, __result_span, __config);
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
    ::cuda::__driver::__memcpyAsync(&__result, __counter.data(), sizeof(size_type), __stream.get());
#  endif // _CCCL_CTK_BELOW(13, 0)
    __stream.sync();
    return __result;
  }

public:
  // Host bulk operations have the same iterator requirements and operation semantics as the
  // corresponding fixed_capacity_map methods. They act on this ref's existing storage.

  //! @brief Erases all elements and synchronizes `__stream`.
  //!
  //! @see fixed_capacity_map::clear
  _CCCL_HOST_API void clear(::cuda::stream_ref __stream) const
  {
    clear_async(__stream);
    __stream.sync();
  }

  //! @brief Asynchronously erases all elements from the referenced storage.
  //!
  //! @see fixed_capacity_map::clear_async
  _CCCL_HOST_API void clear_async(::cuda::stream_ref __stream) const
  {
    if (capacity() == 0)
    {
      return;
    }
    _CCCL_TRY_RUNTIME_API(
      CUB_NS_QUALIFIER::DeviceTransform::Fill,
      "cuco: failed to clear slot storage",
      __impl.storage_ref().data(),
      static_cast<detail::__index_type>(capacity()),
      __impl.empty_slot_sentinel(),
      __stream);
  }

  //! @brief Inserts `[__first, __last)` and returns the number of successful insertions.
  //!
  //! Uses `__mr` for temporary device storage and synchronizes `__stream`.
  //! @see fixed_capacity_map::insert
  template <class _InputIt, class _MemoryResource>
  _CCCL_HOST_API size_type
  insert(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _MemoryResource __mr) const
  {
    return insert_if(__stream, __first, __last, ::cuda::constant_iterator<bool>{true}, ::cuda::std::identity{}, __mr);
  }

  //! @brief Asynchronously inserts `[__first, __last)`.
  //!
  //! @see fixed_capacity_map::insert_async
  template <class _InputIt>
  _CCCL_HOST_API void insert_async(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last) const
  {
    insert_if_async(__stream, __first, __last, ::cuda::constant_iterator<bool>{true}, ::cuda::std::identity{});
  }

  //! @brief Inserts each pair, writes its stored mapped value and insertion status, and synchronizes `__stream`.
  //!
  //! Mapped values must differ from the empty value sentinel. Concurrent operations must
  //! also use `insert_and_find` or `insert_and_find_async`.
  //! @see fixed_capacity_map::insert_and_find
  template <class _InputIt, class _FoundIt, class _InsertedIt>
  _CCCL_HOST_API void insert_and_find(
    ::cuda::stream_ref __stream,
    _InputIt __first,
    _InputIt __last,
    _FoundIt __found_begin,
    _InsertedIt __inserted_begin) const
  {
    insert_and_find_async(__stream, __first, __last, __found_begin, __inserted_begin);
    __stream.sync();
  }

  //! @brief Asynchronously inserts each pair and writes its stored mapped value and insertion status.
  //!
  //! Mapped values must differ from the empty value sentinel. Concurrent operations must
  //! also use `insert_and_find` or `insert_and_find_async`.
  //! @see fixed_capacity_map::insert_and_find_async
  template <class _InputIt, class _FoundIt, class _InsertedIt>
  _CCCL_HOST_API void insert_and_find_async(
    ::cuda::stream_ref __stream,
    _InputIt __first,
    _InputIt __last,
    _FoundIt __found_begin,
    _InsertedIt __inserted_begin) const
  {
    const auto __num_keys = detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    const auto __config = __launch_config(__num_keys);
    ::cuda::launch(
      __stream,
      __config,
      __open_addressing::
        __insert_and_find_n<cg_size, detail::__default_block_size, _InputIt, _FoundIt, _InsertedIt, fixed_capacity_map_ref>,
      __first,
      __num_keys,
      __found_begin,
      __inserted_begin,
      *this);
  }

  //! @brief Inserts each pair selected by `__pred(__stencil[i])` and returns the number inserted.
  //!
  //! Uses `__mr` for temporary device storage and synchronizes `__stream`.
  //! @see fixed_capacity_map::insert_if
  template <class _InputIt, class _StencilIt, class _Predicate, class _MemoryResource>
  _CCCL_HOST_API size_type insert_if(
    ::cuda::stream_ref __stream,
    _InputIt __first,
    _InputIt __last,
    _StencilIt __stencil,
    _Predicate __pred,
    _MemoryResource __mr) const
  {
    const auto __num_keys = detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return 0;
    }

    auto __counter = __make_counter(__stream, __mr);

    const auto __config = __launch_config(__num_keys);
    using __kernel_type =
      void (*)(_InputIt, detail::__index_type, _StencilIt, _Predicate, size_type*, fixed_capacity_map_ref);
    const auto __kernel = static_cast<__kernel_type>(
      __open_addressing::
        __insert_if_n<cg_size, detail::__default_block_size, _InputIt, _StencilIt, _Predicate, fixed_capacity_map_ref>);
    ::cuda::launch(__stream, __config, __kernel, __first, __num_keys, __stencil, __pred, __counter.data(), *this);

    return __read_counter(__counter, __stream);
  }

  //! @brief Asynchronously inserts each pair selected by `__pred(__stencil[i])`.
  //!
  //! @see fixed_capacity_map::insert_if_async
  template <class _InputIt, class _StencilIt, class _Predicate>
  _CCCL_HOST_API void insert_if_async(
    ::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _StencilIt __stencil, _Predicate __pred) const
  {
    const auto __num_keys = detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    if constexpr (cg_size == 1)
    {
      __open_addressing::__insert_if_fn __op{__first, __stencil, __pred, *this};
      _CCCL_TRY_RUNTIME_API(
        CUB_NS_QUALIFIER::DeviceFor::Bulk, "cuco: failed to insert keys", __num_keys, __op, __stream);
    }
    else
    {
      const auto __config = __launch_config(__num_keys);
      using __kernel_type = void (*)(_InputIt, detail::__index_type, _StencilIt, _Predicate, fixed_capacity_map_ref);
      const auto __kernel = static_cast<__kernel_type>(
        __open_addressing::
          __insert_if_n<cg_size, detail::__default_block_size, _InputIt, _StencilIt, _Predicate, fixed_capacity_map_ref>);
      ::cuda::launch(__stream, __config, __kernel, __first, __num_keys, __stencil, __pred, *this);
    }
  }

  //! @brief Inserts or assigns each pair and synchronizes `__stream`.
  //!
  //! @see fixed_capacity_map::insert_or_assign
  template <class _InputIt>
  _CCCL_HOST_API void insert_or_assign(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last) const
  {
    insert_or_assign_async(__stream, __first, __last);
    __stream.sync();
  }

  //! @brief Asynchronously inserts or assigns each pair.
  //!
  //! @see fixed_capacity_map::insert_or_assign_async
  template <class _InputIt>
  _CCCL_HOST_API void insert_or_assign_async(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last) const
  {
    const auto __num_keys = detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }
    const auto __config = __launch_config(__num_keys);
    const auto& __kernel =
      __open_addressing::__insert_or_assign_n<cg_size, detail::__default_block_size, _InputIt, fixed_capacity_map_ref>;
    ::cuda::launch(__stream, __config, __kernel, __first, __num_keys, *this);
  }

  //! @brief Writes whether each key is present and synchronizes `__stream`.
  //!
  //! @see fixed_capacity_map::contains
  template <class _InputIt, class _OutputIt>
  _CCCL_HOST_API void
  contains(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _OutputIt __output_begin) const
  {
    contains_async(__stream, __first, __last, __output_begin);
    __stream.sync();
  }

  //! @brief Asynchronously writes whether each key is present.
  //!
  //! @see fixed_capacity_map::contains_async
  template <class _InputIt, class _OutputIt>
  _CCCL_HOST_API void
  contains_async(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _OutputIt __output_begin) const
  {
    contains_if_async(
      __stream, __first, __last, ::cuda::constant_iterator<bool>{true}, ::cuda::std::identity{}, __output_begin);
  }

  //! @brief Queries the selected keys, writes false for unselected keys, and synchronizes `__stream`.
  //!
  //! @see fixed_capacity_map::contains_if
  template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt>
  _CCCL_HOST_API void contains_if(
    ::cuda::stream_ref __stream,
    _InputIt __first,
    _InputIt __last,
    _StencilIt __stencil,
    _Predicate __pred,
    _OutputIt __output_begin) const
  {
    contains_if_async(__stream, __first, __last, __stencil, __pred, __output_begin);
    __stream.sync();
  }

  //! @brief Asynchronously queries the selected keys and writes false for unselected keys.
  //!
  //! @see fixed_capacity_map::contains_if_async
  template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt>
  _CCCL_HOST_API void contains_if_async(
    ::cuda::stream_ref __stream,
    _InputIt __first,
    _InputIt __last,
    _StencilIt __stencil,
    _Predicate __pred,
    _OutputIt __output_begin) const
  {
    const auto __num_keys = detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    if constexpr (cg_size == 1)
    {
      __open_addressing::__contains_if_fn __op{__first, __stencil, __pred, __output_begin, *this};
      _CCCL_TRY_RUNTIME_API(CUB_NS_QUALIFIER::DeviceFor::Bulk, "cuco: failed to query keys", __num_keys, __op, __stream);
    }
    else
    {
      const auto __config  = __launch_config(__num_keys);
      const auto& __kernel = __open_addressing::__contains_if_n<
        cg_size,
        detail::__default_block_size,
        _InputIt,
        _StencilIt,
        _Predicate,
        _OutputIt,
        fixed_capacity_map_ref>;
      ::cuda::launch(__stream, __config, __kernel, __first, __num_keys, __stencil, __pred, __output_begin, *this);
    }
  }

  //! @brief Writes each key's mapped value or the empty value sentinel and synchronizes `__stream`.
  //!
  //! @see fixed_capacity_map::find
  template <class _InputIt, class _OutputIt>
  _CCCL_HOST_API void
  find(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _OutputIt __output_begin) const
  {
    find_async(__stream, __first, __last, __output_begin);
    __stream.sync();
  }

  //! @brief Asynchronously writes each key's mapped value or the empty value sentinel.
  //!
  //! @see fixed_capacity_map::find_async
  template <class _InputIt, class _OutputIt>
  _CCCL_HOST_API void
  find_async(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _OutputIt __output_begin) const
  {
    find_if_async(
      __stream, __first, __last, ::cuda::constant_iterator<bool>{true}, ::cuda::std::identity{}, __output_begin);
  }

  //! @brief Writes mapped values for selected keys or the empty value sentinel and synchronizes `__stream`.
  //!
  //! @see fixed_capacity_map::find_if
  template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt>
  _CCCL_HOST_API void find_if(
    ::cuda::stream_ref __stream,
    _InputIt __first,
    _InputIt __last,
    _StencilIt __stencil,
    _Predicate __pred,
    _OutputIt __output_begin) const
  {
    find_if_async(__stream, __first, __last, __stencil, __pred, __output_begin);
    __stream.sync();
  }

  //! @brief Asynchronously writes mapped values for selected keys, or the empty value sentinel otherwise.
  //!
  //! @see fixed_capacity_map::find_if_async
  template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt>
  _CCCL_HOST_API void find_if_async(
    ::cuda::stream_ref __stream,
    _InputIt __first,
    _InputIt __last,
    _StencilIt __stencil,
    _Predicate __pred,
    _OutputIt __output_begin) const
  {
    const auto __num_keys = detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    const auto __config  = __launch_config(__num_keys);
    const auto& __kernel = __open_addressing::__find_if_n<
      cg_size,
      detail::__default_block_size,
      _InputIt,
      _StencilIt,
      _Predicate,
      _OutputIt,
      fixed_capacity_map_ref>;
    ::cuda::launch(__stream, __config, __kernel, __first, __num_keys, __stencil, __pred, __output_begin, *this);
  }

  //! @brief Applies the callback to a copy of each matching slot and synchronizes `__stream`.
  //!
  //! @see fixed_capacity_map::for_each
  template <class _InputIt, class _CallbackOp>
  _CCCL_HOST_API void
  for_each(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _CallbackOp __callback_op) const
  {
    for_each_async(__stream, __first, __last, __callback_op);
    __stream.sync();
  }

  //! @brief Asynchronously applies the callback to a copy of each matching slot.
  //!
  //! @see fixed_capacity_map::for_each_async
  template <class _InputIt, class _CallbackOp>
  _CCCL_HOST_API void
  for_each_async(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _CallbackOp __callback_op) const
  {
    const auto __num_keys = detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    if constexpr (cg_size == 1)
    {
      __open_addressing::__for_each_fn __op{__first, __callback_op, *this};
      _CCCL_TRY_RUNTIME_API(CUB_NS_QUALIFIER::DeviceFor::Bulk, "cuco: failed to query keys", __num_keys, __op, __stream);
    }
    else
    {
      const auto __config  = __launch_config(__num_keys);
      const auto& __kernel = __open_addressing::
        __for_each_n<cg_size, detail::__default_block_size, _InputIt, _CallbackOp, fixed_capacity_map_ref>;
      ::cuda::launch(__stream, __config, __kernel, __first, __num_keys, __callback_op, *this);
    }
  }

  //! @brief Copies all occupied key-value pairs to the output ranges and returns their ends.
  //!
  //! Uses `__mr` for temporary device storage and synchronizes `__stream`.
  //! @see fixed_capacity_map::retrieve_all
  template <class _KeyOutputIt, class _ValueOutputIt, class _MemoryResource>
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::pair<_KeyOutputIt, _ValueOutputIt> retrieve_all(
    ::cuda::stream_ref __stream, _KeyOutputIt __keys_out, _ValueOutputIt __values_out, _MemoryResource __mr) const
  {
    auto __counter           = __make_counter(__stream, __mr);
    const auto __input_begin = ::cuda::make_transform_iterator(
      ::cuda::counting_iterator<size_type>{0},
      __open_addressing::__get_slot<true, __storage_ref_type>{__impl.storage_ref()});
    const auto __is_filled =
      __open_addressing::__slot_is_filled<true, key_type>{empty_key_sentinel(), erased_key_sentinel()};
    const auto __env          = ::cuda::std::execution::env{__stream, __mr};
    const auto __output_begin = ::cuda::make_zip_iterator(__keys_out, __values_out);
    _CCCL_TRY_RUNTIME_API(
      CUB_NS_QUALIFIER::DeviceSelect::If,
      "cuco: failed to retrieve all elements",
      __input_begin,
      __output_begin,
      __counter.data(),
      capacity(),
      __is_filled,
      __env);
    const auto __num_out = __read_counter(__counter, __stream);
    return {__keys_out + __num_out, __values_out + __num_out};
  }

  //! @brief Counts the occupied slots in the referenced storage.
  //!
  //! Uses `__mr` for temporary device storage and synchronizes `__stream`.
  //! @see fixed_capacity_map::size
  template <class _MemoryResource>
  [[nodiscard]] _CCCL_HOST_API size_type size(::cuda::stream_ref __stream, _MemoryResource __mr) const
  {
    auto __counter           = __make_counter(__stream, __mr);
    const auto __input_begin = ::cuda::make_transform_iterator(
      ::cuda::counting_iterator<size_type>{0},
      __open_addressing::__get_slot<true, __storage_ref_type>{__impl.storage_ref()});
    const auto __is_filled =
      __open_addressing::__slot_is_filled<true, key_type>{empty_key_sentinel(), erased_key_sentinel()};
    const auto __env = ::cuda::std::execution::env{__stream, __mr};
    _CCCL_TRY_RUNTIME_API(
      CUB_NS_QUALIFIER::DeviceReduce::TransformReduce,
      "cuco: failed to get the number of elements",
      __input_begin,
      __counter.data(),
      capacity(),
      ::cuda::std::plus<size_type>{},
      __is_filled,
      size_type{0},
      __env);
    return __read_counter(__counter, __stream);
  }

#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()
  // ===== Storage operations =====

  //! @brief Cooperatively initializes the map's slot storage with the empty slot sentinel.
  //!
  //! This function turns a ref constructed over raw (uninitialized) memory, e.g. a shared-memory
  //! array, into an empty, ready-to-use map. It can also be used to clear an existing map.
  //!
  //! @note This is a group-collective operation: all threads of `__group` must call it. The group
  //! is synchronized before this function returns.
  //! @note No other operation may access the map concurrently. When clearing a map after earlier
  //! operations, the caller must ensure those operations have completed before calling this function.
  //!
  //! @tparam _Group Cooperative group type
  //!
  //! @param[in] __group The cooperative group used to initialize the storage
  template <class _Group>
  _CCCL_DEVICE_API constexpr void initialize(_Group __group) noexcept
  {
    __impl.initialize(__group);
  }

  //! @brief Cooperatively copies the map's slots to `__slots` and returns a new ref operating on
  //! the copy.
  //!
  //! This function is intended to create shared-memory copies of small maps, although any device
  //! memory can be used as well. The thread scope of the returned ref can be set via `_NewScope`,
  //! e.g. `__ref.make_copy<::cuda::thread_scope_block>(__block, __slots)` for a copy that is only
  //! accessed by threads of a single block.
  //!
  //! @note This is a group-collective operation: all threads of `__group` must call it with the
  //! same `__slots`. The group is synchronized before this function returns. The source map must
  //! not be modified concurrently, and the destination must not be accessed concurrently.
  //! @note The source and destination storage ranges must not overlap.
  //! @note `value_type` must be trivially copyable.
  //!
  //! @tparam _NewScope Thread scope of the returned ref (defaults to this ref's scope)
  //! @tparam _Group Cooperative group type
  //!
  //! @param[in] __group The cooperative group used to perform the copy
  //! @param[out] __slots Span over the target slot storage; must contain exactly `capacity()` slots
  //!
  //! @return A new ref with thread scope `_NewScope` operating on `__slots`
  template <::cuda::thread_scope _NewScope = _Scope, class _Group>
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto make_copy(_Group __group, storage_span_type __slots) const noexcept
    -> fixed_capacity_map_ref<_Key, _Tp, _NewScope, _KeyEqual, _ProbingScheme, _BucketSize, _Capacity>
  {
    if constexpr (_Capacity == ::cuda::std::dynamic_extent)
    {
      _CCCL_ASSERT(__slots.size() == capacity(), "destination storage size must equal the map capacity");
    }
    __impl.make_copy(__group, __slots.data());
    using __copy_ref_type =
      fixed_capacity_map_ref<_Key, _Tp, _NewScope, _KeyEqual, _ProbingScheme, _BucketSize, _Capacity>;
    return detail::__bitwise_compare(empty_key_sentinel(), erased_key_sentinel())
           ? __copy_ref_type{empty_key<_Key>{empty_key_sentinel()},
                             empty_value<_Tp>{empty_value_sentinel()},
                             key_eq(),
                             probing_scheme(),
                             __slots}
           : __copy_ref_type{
               empty_key<_Key>{empty_key_sentinel()},
               empty_value<_Tp>{empty_value_sentinel()},
               erased_key<_Key>{erased_key_sentinel()},
               key_eq(),
               probing_scheme(),
               __slots};
  }

  // ===== Insert operations =====

  //! @brief Inserts a key-value pair.
  //!
  //! @param __value The key-value pair to insert
  //!
  //! @return `true` if the pair was inserted, `false` if the key already exists
  _CCCL_DEVICE_API bool insert(value_type __value) noexcept
  {
    return __impl.insert(__value);
  }

  //! @brief Inserts a key-value pair using a cooperative group.
  //!
  //! @tparam _ParentCG Parent cooperative group type
  //!
  //! @param __group The cooperative group used for this operation
  //! @param __value The key-value pair to insert
  //!
  //! @return `true` if the pair was inserted, `false` if the key already exists
  template <class _ParentCG>
  _CCCL_DEVICE_API bool
  insert(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, value_type __value) noexcept
  {
    return __impl.insert(__group, __value);
  }

  //! @brief Inserts a key-value pair and returns its slot.
  //!
  //! If an equivalent key is already present, returns an iterator to the existing pair and `false`.
  //! If insertion succeeds, returns an iterator to the inserted pair and `true`.
  //! If no slot is available, returns `end()` and `false`.
  //!
  //! @note Concurrent calls for the same key return the payload of the insertion that succeeds.
  //! @pre Input and stored mapped values must not equal `empty_value_sentinel()`.
  //! @pre Concurrent operations on this map must also use `insert_and_find`.
  //!
  //! @tparam _Value Input type convertible to `value_type`
  //!
  //! @param[in] __value The key-value pair to insert
  //!
  //! @return The pair's iterator and whether insertion succeeded, or `{end(), false}` if the map is full
  template <class _Value>
  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::pair<iterator, bool> insert_and_find(_Value __value) noexcept
  {
    return __impl.insert_and_find(__value);
  }

  //! @brief Cooperative-group variant of `insert_and_find`.
  //!
  //! If an equivalent key is already present, returns an iterator to the existing pair and `false`.
  //! If insertion succeeds, returns an iterator to the inserted pair and `true`.
  //! If no slot is available, returns `end()` and `false`.
  //!
  //! @note Concurrent calls for the same key return the payload of the insertion that succeeds.
  //! @pre Input and stored mapped values must not equal `empty_value_sentinel()`.
  //! @pre Concurrent operations on this map must also use `insert_and_find`.
  //!
  //! @tparam _Value Input type convertible to `value_type`
  //! @tparam _ParentCG Parent cooperative group type
  //!
  //! @param[in] __group The cooperative group used for this operation
  //! @param[in] __value The key-value pair to insert
  //!
  //! @return The pair's iterator and whether insertion succeeded, or `{end(), false}` if the map is full
  template <class _Value, class _ParentCG>
  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::pair<iterator, bool>
  insert_and_find(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, _Value __value) noexcept
  {
    return __impl.insert_and_find(__group, __value);
  }

  //! @brief Inserts `__value` if its key is absent, otherwise assigns its mapped value.
  //!
  //! @note Requires `cg_size == 1`. Concurrent assignments to the same key leave an
  //! unspecified one of the assigned values. Concurrent lookup and modification are unsupported.
  //!
  //! @param[in] __value The key-value pair to insert or assign
  _CCCL_DEVICE_API void insert_or_assign(value_type __value) noexcept
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");
    // An erased slot can precede an existing key in the probe sequence. Find that key before
    // claiming a reusable slot, otherwise assignment could create a duplicate.
    if (!detail::__bitwise_compare(empty_key_sentinel(), erased_key_sentinel()))
    {
      if (const auto __slot = find(__value.first); __slot != end())
      {
        ::cuda::atomic_ref<mapped_type, _Scope>{__slot->second}.store(__value.second, ::cuda::memory_order_relaxed);
        return;
      }
    }
    const auto __storage = __impl.storage_ref();
    auto __iter = probing_scheme().template make_iterator<bucket_size>(__value.first, __storage.capacity_extent());
    const auto __initial = *__iter;
    do
    {
      const auto __slots = __storage[*__iter];
      for (int __i = 0; __i < bucket_size; ++__i)
      {
        const auto __state =
          __impl.predicate().template operator()<detail::__is_insert::__yes>(__value.first, __slots[__i].first);
        if (__state == detail::__equal_result::__equal)
        {
          ::cuda::atomic_ref<mapped_type, _Scope>{__slots[__i].second}.store(
            __value.second, ::cuda::memory_order_relaxed);
          return;
        }
        if (__state == detail::__equal_result::__available && __attempt_insert_or_assign(&__slots[__i], __value))
        {
          return;
        }
      }
      ++__iter;
    } while (*__iter != __initial);
  }

  //! @brief Inserts or assigns a key-value pair using a cooperative group.
  //!
  //! @note All threads in `__group` must participate with the same value. Concurrent assignments
  //! to the same key leave an unspecified one of the assigned values.
  //!
  //! @tparam _ParentCG Parent cooperative group type
  //! @param[in] __group The cooperative group used for this operation
  //! @param[in] __value The key-value pair to insert or assign
  template <class _ParentCG>
  _CCCL_DEVICE_API void
  insert_or_assign(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, value_type __value) noexcept
  {
    // Search past erased slots before reusing one, as in the scalar overload.
    if (!detail::__bitwise_compare(empty_key_sentinel(), erased_key_sentinel()))
    {
      if (const auto __slot = find(__group, __value.first); __slot != end())
      {
        if (__group.thread_rank() == 0)
        {
          ::cuda::atomic_ref<mapped_type, _Scope>{__slot->second}.store(__value.second, ::cuda::memory_order_relaxed);
        }
        __group.sync();
        return;
      }
    }
    const auto __storage = __impl.storage_ref();
    auto __iter =
      probing_scheme().template make_iterator<bucket_size>(__group, __value.first, __storage.capacity_extent());
    const auto __initial = *__iter;
    while (true)
    {
      const auto [__state, __index] = __impl.__find_insert_slot(__value.first, __storage[*__iter]);
      const auto __equal            = __group.ballot(__state == detail::__equal_result::__equal);
      if (__equal)
      {
        const auto __lane = __ffs(__equal) - 1;
        if (__group.thread_rank() == __lane)
        {
          auto* __slot = __impl.__get_slot_ptr(*__iter, __index);
          ::cuda::atomic_ref<mapped_type, _Scope>{__slot->second}.store(__value.second, ::cuda::memory_order_relaxed);
        }
        __group.sync();
        return;
      }
      const auto __available = __group.ballot(__state == detail::__equal_result::__available);
      if (__available)
      {
        const auto __lane = __ffs(__available) - 1;
        bool __success    = false;
        if (__group.thread_rank() == __lane)
        {
          __success = __attempt_insert_or_assign(__impl.__get_slot_ptr(*__iter, __index), __value);
        }
        if (__group.shfl(__success, __lane))
        {
          return;
        }
      }
      else
      {
        ++__iter;
        if (*__iter == __initial)
        {
          return;
        }
      }
    }
  }

  // ===== Lookup operations =====

  //! @brief Checks if a key exists in the map.
  //!
  //! @param __key The key to search for
  //!
  //! @return `true` if the key is found
  template <class _ProbeKey = key_type>
  [[nodiscard]] _CCCL_DEVICE_API bool contains(_ProbeKey __key) const noexcept
  {
    return __impl.contains(__key);
  }

  //! @brief Cooperative-group variant of `contains`.
  //!
  //! @tparam _ParentCG Parent cooperative group type
  //! @tparam _ProbeKey Probe key type (defaults to `key_type`)
  //!
  //! @param __group Cooperative group of size `cg_size` performing this lookup
  //! @param __key The key to search for
  //!
  //! @return `true` if the key is found
  template <class _ParentCG, class _ProbeKey = key_type>
  [[nodiscard]] _CCCL_DEVICE_API bool
  contains(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, _ProbeKey __key) const noexcept
  {
    return __impl.contains(__group, __key);
  }

  //! @brief Finds the slot associated with a key.
  //!
  //! @tparam _ProbeKey Probe key type (defaults to `key_type`)
  //!
  //! @param __key The key to search for
  //!
  //! @return An iterator to the slot holding `__key`, or `end()` if the key is not found
  template <class _ProbeKey = key_type>
  [[nodiscard]] _CCCL_DEVICE_API iterator find(_ProbeKey __key) const noexcept
  {
    return __impl.find(__key);
  }

  //! @brief Cooperative-group variant of `find`.
  //!
  //! @tparam _ParentCG Parent cooperative group type
  //! @tparam _ProbeKey Probe key type (defaults to `key_type`)
  //!
  //! @param __group Cooperative group of size `cg_size` performing this lookup
  //! @param __key The key to search for
  //!
  //! @return An iterator to the slot holding `__key`, or `end()` if the key is not found
  template <class _ParentCG, class _ProbeKey = key_type>
  [[nodiscard]] _CCCL_DEVICE_API iterator
  find(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, _ProbeKey __key) const noexcept
  {
    return __impl.find(__group, __key);
  }

  //! @brief Applies `__callback_op` to a copy of every slot whose key is equivalent to `__key`.
  //!
  //! @note The return value of `__callback_op`, if any, is ignored.
  //!
  //! @tparam _ProbeKey Probe key type
  //! @tparam _CallbackOp Unary callable invocable with `value_type`
  //!
  //! @param __key The key to search for
  //! @param __callback_op Function to apply to every matching slot
  template <class _ProbeKey, class _CallbackOp>
  _CCCL_DEVICE_API void for_each(_ProbeKey __key, _CallbackOp&& __callback_op) const noexcept
  {
    __impl.for_each(__key, ::cuda::std::forward<_CallbackOp>(__callback_op));
  }

  //! @brief Cooperative-group variant of `for_each`.
  //!
  //! @note Any thread in `__group` may invoke the callback. If multiple threads find a match, each
  //! of them invokes the callback with its own matching slot.
  //!
  //! @note Synchronizing `__group` inside `__callback_op` is undefined behavior.
  //!
  //! @note The return value of `__callback_op`, if any, is ignored.
  //!
  //! @tparam _ParentCG Parent cooperative group type
  //! @tparam _ProbeKey Probe key type
  //! @tparam _CallbackOp Unary callable invocable with `value_type`
  //!
  //! @param __group Cooperative group of size `cg_size` performing this operation
  //! @param __key The key to search for
  //! @param __callback_op Function to apply to every matching slot
  template <class _ParentCG, class _ProbeKey, class _CallbackOp>
  _CCCL_DEVICE_API void for_each(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group,
                                 _ProbeKey __key,
                                 _CallbackOp&& __callback_op) const noexcept
  {
    __impl.for_each(__group, __key, ::cuda::std::forward<_CallbackOp>(__callback_op));
  }

private:
  //! @brief Claims an empty or erased slot, or assigns the payload of a competing insertion with the same key.
  [[nodiscard]] _CCCL_DEVICE_API bool __attempt_insert_or_assign(value_type* __slot, value_type __value) noexcept
  {
    auto __expected = empty_key_sentinel();
    const ::cuda::atomic_ref<key_type, _Scope> __key_ref{__slot->first};
    bool __inserted = __key_ref.compare_exchange_strong(__expected, __value.first, ::cuda::memory_order_relaxed);
    if (!__inserted && detail::__bitwise_compare(__expected, erased_key_sentinel()))
    {
      __inserted = __key_ref.compare_exchange_strong(__expected, __value.first, ::cuda::memory_order_relaxed);
    }
    if (__inserted
        || __impl.predicate().template operator()<detail::__is_insert::__no>(__value.first, __expected)
             == detail::__equal_result::__equal)
    {
      ::cuda::atomic_ref<mapped_type, _Scope>{__slot->second}.store(__value.second, ::cuda::memory_order_relaxed);
      return true;
    }
    return false;
  }

#endif // _CCCL_CUDA_COMPILATION()
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_FIXED_CAPACITY_MAP_REF_CUH
