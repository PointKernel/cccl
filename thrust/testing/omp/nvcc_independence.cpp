#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system_error.h>
#include <thrust/transform.h>

#include <unittest/unittest.h>

void TestNvccIndependenceTransform()
{
  using T     = int;
  const int n = 10;

  thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(n);
  thrust::device_vector<T> d_output(n);

  thrust::transform(h_input.begin(), h_input.end(), h_output.begin(), ::cuda::std::negate<T>());
  thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), ::cuda::std::negate<T>());

  ASSERT_EQUAL(h_output, d_output);
}
DECLARE_UNITTEST(TestNvccIndependenceTransform);

void TestNvccIndependenceReduce()
{
  using T     = int;
  const int n = 10;

  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  T init = 13;

  T h_result = thrust::reduce(h_data.begin(), h_data.end(), init);
  T d_result = thrust::reduce(d_data.begin(), d_data.end(), init);

  ASSERT_ALMOST_EQUAL(h_result, d_result);
}
DECLARE_UNITTEST(TestNvccIndependenceReduce);

void TestNvccIndependenceExclusiveScan()
{
  using T     = int;
  const int n = 10;

  thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(n);
  thrust::device_vector<T> d_output(n);

  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
  thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
  ASSERT_EQUAL(d_output, h_output);
}
DECLARE_UNITTEST(TestNvccIndependenceExclusiveScan);

void TestNvccIndependenceSort()
{
  using T     = int;
  const int n = 10;

  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::sort(h_data.begin(), h_data.end(), ::cuda::std::less<T>());
  thrust::sort(d_data.begin(), d_data.end(), ::cuda::std::less<T>());

  ASSERT_EQUAL(h_data, d_data);
}
DECLARE_UNITTEST(TestNvccIndependenceSort);
