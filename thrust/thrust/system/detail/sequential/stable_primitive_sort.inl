/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/type_traits.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/detail/sequential/partition.h>
#include <thrust/system/detail/sequential/stable_primitive_sort.h>
#include <thrust/system/detail/sequential/stable_radix_sort.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{
namespace stable_primitive_sort_detail
{

template <typename Iterator>
struct enable_if_bool_sort
    : ::cuda::std::enable_if<::cuda::std::is_same<bool, thrust::detail::it_value_t<Iterator>>::value>
{};

template <typename Iterator>
struct disable_if_bool_sort
    : thrust::detail::disable_if<::cuda::std::is_same<bool, thrust::detail::it_value_t<Iterator>>::value>
{};

template <typename DerivedPolicy, typename RandomAccessIterator>
typename enable_if_bool_sort<RandomAccessIterator>::type _CCCL_HOST_DEVICE stable_primitive_sort(
  sequential::execution_policy<DerivedPolicy>& exec, RandomAccessIterator first, RandomAccessIterator last)
{
  // use stable_partition if we're sorting bool
  // stable_partition puts true values first, so we need to logical_not
  sequential::stable_partition(exec, first, last, ::cuda::std::logical_not<bool>());
}

template <typename DerivedPolicy, typename RandomAccessIterator>
typename disable_if_bool_sort<RandomAccessIterator>::type _CCCL_HOST_DEVICE stable_primitive_sort(
  sequential::execution_policy<DerivedPolicy>& exec, RandomAccessIterator first, RandomAccessIterator last)
{
  // call stable_radix_sort
  sequential::stable_radix_sort(exec, first, last);
}

struct logical_not_first
{
  template <typename Tuple>
  _CCCL_HOST_DEVICE bool operator()(Tuple t)
  {
    return !thrust::get<0>(t);
  }
};

template <typename DerivedPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2>
typename enable_if_bool_sort<RandomAccessIterator1>::type _CCCL_HOST_DEVICE stable_primitive_sort_by_key(
  sequential::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator1 keys_first,
  RandomAccessIterator1 keys_last,
  RandomAccessIterator2 values_first)
{
  // use stable_partition if we're sorting bool
  // stable_partition puts true values first, so we need to logical_not
  sequential::stable_partition(
    exec,
    thrust::make_zip_iterator(keys_first, values_first),
    thrust::make_zip_iterator(keys_last, values_first),
    logical_not_first());
}

template <typename DerivedPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2>
typename disable_if_bool_sort<RandomAccessIterator1>::type _CCCL_HOST_DEVICE stable_primitive_sort_by_key(
  sequential::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator1 keys_first,
  RandomAccessIterator1 keys_last,
  RandomAccessIterator2 values_first)
{
  // call stable_radix_sort_by_key
  sequential::stable_radix_sort_by_key(exec, keys_first, keys_last, values_first);
}

} // namespace stable_primitive_sort_detail

template <typename DerivedPolicy, typename RandomAccessIterator>
_CCCL_HOST_DEVICE void stable_primitive_sort(
  sequential::execution_policy<DerivedPolicy>& exec, RandomAccessIterator first, RandomAccessIterator last)
{
  stable_primitive_sort_detail::stable_primitive_sort(exec, first, last);
}

template <typename DerivedPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2>
_CCCL_HOST_DEVICE void stable_primitive_sort_by_key(
  sequential::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator1 keys_first,
  RandomAccessIterator1 keys_last,
  RandomAccessIterator2 values_first)
{
  stable_primitive_sort_detail::stable_primitive_sort_by_key(exec, keys_first, keys_last, values_first);
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END
