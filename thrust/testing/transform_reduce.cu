#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/retag.h>
#include <thrust/transform_reduce.h>

#include <unittest/unittest.h>

template <typename InputIterator, typename UnaryFunction, typename OutputType, typename BinaryFunction>
OutputType
transform_reduce(my_system& system, InputIterator, InputIterator, UnaryFunction, OutputType init, BinaryFunction)
{
  system.validate_dispatch();
  return init;
}

void TestTransformReduceDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::transform_reduce(sys, vec.begin(), vec.begin(), 0, 0, 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestTransformReduceDispatchExplicit);

template <typename InputIterator, typename UnaryFunction, typename OutputType, typename BinaryFunction>
OutputType transform_reduce(my_tag, InputIterator first, InputIterator, UnaryFunction, OutputType init, BinaryFunction)
{
  *first = 13;
  return init;
}

void TestTransformReduceDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::transform_reduce(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0, 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestTransformReduceDispatchImplicit);

template <class Vector>
void TestTransformReduceSimple()
{
  using T = typename Vector::value_type;

  Vector data{1, -2, 3};

  T init   = 10;
  T result = thrust::transform_reduce(data.begin(), data.end(), ::cuda::std::negate<T>(), init, ::cuda::std::plus<T>());

  ASSERT_EQUAL(result, 8);
}
DECLARE_VECTOR_UNITTEST(TestTransformReduceSimple);

template <typename T>
void TestTransformReduce(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  T init = 13;

  T cpu_result =
    thrust::transform_reduce(h_data.begin(), h_data.end(), ::cuda::std::negate<T>(), init, ::cuda::std::plus<T>());
  T gpu_result =
    thrust::transform_reduce(d_data.begin(), d_data.end(), ::cuda::std::negate<T>(), init, ::cuda::std::plus<T>());

  ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestTransformReduce);

template <typename T>
void TestTransformReduceFromConst(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  T init = 13;

  T cpu_result =
    thrust::transform_reduce(h_data.cbegin(), h_data.cend(), ::cuda::std::negate<T>(), init, ::cuda::std::plus<T>());
  T gpu_result =
    thrust::transform_reduce(d_data.cbegin(), d_data.cend(), ::cuda::std::negate<T>(), init, ::cuda::std::plus<T>());

  ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestTransformReduceFromConst);

template <class Vector>
void TestTransformReduceCountingIterator()
{
  using T     = typename Vector::value_type;
  using space = typename thrust::iterator_system<typename Vector::iterator>::type;

  thrust::counting_iterator<T, space> first(1);

  T result = thrust::transform_reduce(first, first + 3, ::cuda::std::negate<short>(), 0, ::cuda::std::plus<short>());

  ASSERT_EQUAL(result, -6);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestTransformReduceCountingIterator);
