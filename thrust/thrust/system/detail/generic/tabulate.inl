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
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/tabulate.h>
#include <thrust/transform.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename ForwardIterator, typename UnaryOperation>
_CCCL_HOST_DEVICE void tabulate(
  thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, UnaryOperation unary_op)
{
  using difference_type = thrust::detail::it_difference_t<ForwardIterator>;

  // by default, counting_iterator uses a 64b difference_type on 32b platforms to avoid overflowing its counter.
  // this causes problems when a zip_iterator is created in transform's implementation -- ForwardIterator is
  // incremented by a 64b difference_type and some compilers warn
  // to avoid this, specify the counting_iterator's difference_type to be the same as ForwardIterator's.
  thrust::counting_iterator<difference_type, thrust::use_default, thrust::use_default, difference_type> iter(0);

  thrust::transform(exec, iter, iter + ::cuda::std::distance(first, last), first, unary_op);
} // end tabulate()

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END
