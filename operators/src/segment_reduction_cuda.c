#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <THC/THC.h>

#include "cuda/segment_reduction.h"

extern THCState *state;



int unsorted_segment_sum_forward_gpu(
  const THCudaTensor* data_cuda, const THCudaLongTensor* segment_ids_cuda, const int* data_shape, THCudaTensor* output_cuda) {

  cudaStream_t stream = THCState_getCurrentStream(state);

  float* data        = THCudaTensor_data(state, data_cuda);
  long*  segment_ids = THCudaLongTensor_data(state, segment_ids_cuda);
  float* output      = THCudaTensor_data(state, output_cuda);

  unsorted_segment_sum_forward_gpu_kernel_launcher(
    stream, data, segment_ids, data_shape, output);

  return 1;
}

int unsorted_segment_sum_backward_gpu(
  const THCudaTensor* grad_output_cuda, const THCudaLongTensor* segment_ids_cuda, const int* data_shape, THCudaTensor* grad_data_cuda) {

  cudaStream_t stream = THCState_getCurrentStream(state);

  float* grad_output = THCudaTensor_data(state, grad_output_cuda);
  long*  segment_ids = THCudaLongTensor_data(state, segment_ids_cuda);
  float* grad_data   = THCudaTensor_data(state, grad_data_cuda);

  unsorted_segment_sum_backward_gpu_kernel_launcher(
    stream, grad_output, segment_ids, data_shape, grad_data);

  return 1;
}
