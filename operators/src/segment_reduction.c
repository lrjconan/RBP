#include <TH/TH.h>

int unsorted_segment_sum_forward(
  const THFloatTensor* data, const THLongTensor* segment_ids, const int* data_shape, THFloatTensor* output) {

  float* data_ptr        = THFloatTensor_data(data);
  long*  segment_ids_ptr = THLongTensor_data(segment_ids);
  float* output_ptr      = THFloatTensor_data(output);
  int dim_0 = data_shape[0];
  int dim_1 = data_shape[1];
  int dim_2 = data_shape[2];

  for(int ii = 0; ii < dim_0; ++ii)
  {
    for(int jj = 0; jj < dim_1; ++jj)
    {
      int output_idx = segment_ids_ptr[jj];

      for(int kk = 0; kk < dim_2; ++kk)
      {
        output_ptr[ii * dim_1 * dim_2 + output_idx * dim_2 + kk] += data_ptr[ii * dim_1 * dim_2 + jj * dim_2 + kk];
      }
    }
  }

  return 1;
}

int unsorted_segment_sum_backward(
  const THFloatTensor* grad_output, const THLongTensor* segment_ids, const int* data_shape, THFloatTensor* grad_data) {

  float* grad_output_ptr = THFloatTensor_data(grad_output);
  long*  segment_ids_ptr = THLongTensor_data(segment_ids);
  float* grad_data_ptr   = THFloatTensor_data(grad_data);

  int dim_0 = data_shape[0];
  int dim_1 = data_shape[1];
  int dim_2 = data_shape[2];

  for(int ii = 0; ii < dim_0; ++ii)
  {
    for(int jj = 0; jj < dim_1; ++jj)
    {
      int output_idx = segment_ids_ptr[jj];

      for(int kk = 0; kk < dim_2; ++kk)
      {
        grad_data_ptr[ii * dim_1 * dim_2 + jj * dim_2 + kk] = grad_output_ptr[ii * dim_1 * dim_2 + output_idx * dim_2 + kk];
      }
    }
  }

  return 1;
}
