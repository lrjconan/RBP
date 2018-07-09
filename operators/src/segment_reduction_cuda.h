int unsorted_segment_sum_forward_gpu(
  const THCudaTensor* data_cuda, const THCudaLongTensor* segment_ids_cuda, const int* data_shape, THCudaTensor* output_cuda);

int unsorted_segment_sum_backward_gpu(
  const THCudaTensor* grad_output_cuda, const THCudaLongTensor* segment_ids_cuda, const int* data_shape, THCudaTensor* grad_data_cuda);

