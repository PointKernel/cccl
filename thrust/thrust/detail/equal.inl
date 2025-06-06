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
#include <thrust/equal.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/adl/equal.h>
#include <thrust/system/detail/generic/equal.h>
#include <thrust/system/detail/generic/select_system.h>

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename System, typename InputIterator1, typename InputIterator2>
_CCCL_HOST_DEVICE bool
equal(const thrust::detail::execution_policy_base<System>& system,
      InputIterator1 first1,
      InputIterator1 last1,
      InputIterator2 first2)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::equal");
  using thrust::system::detail::generic::equal;
  return equal(thrust::detail::derived_cast(thrust::detail::strip_const(system)), first1, last1, first2);
} // end equal()

_CCCL_EXEC_CHECK_DISABLE
template <typename System, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
_CCCL_HOST_DEVICE bool
equal(const thrust::detail::execution_policy_base<System>& system,
      InputIterator1 first1,
      InputIterator1 last1,
      InputIterator2 first2,
      BinaryPredicate binary_pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::equal");
  using thrust::system::detail::generic::equal;
  return equal(thrust::detail::derived_cast(thrust::detail::strip_const(system)), first1, last1, first2, binary_pred);
} // end equal()

template <typename InputIterator1, typename InputIterator2>
bool equal(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::equal");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;

  System1 system1;
  System2 system2;

  return thrust::equal(select_system(system1, system2), first1, last1, first2);
}

template <typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
bool equal(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate binary_pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::equal");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;

  System1 system1;
  System2 system2;

  return thrust::equal(select_system(system1, system2), first1, last1, first2, binary_pred);
}

THRUST_NAMESPACE_END
