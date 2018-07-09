int unsorted_segment_sum_forward(
  const THFloatTensor* data_cuda, const THLongTensor* segment_ids_cuda, const int* data_shape, THFloatTensor* output_cuda);

int unsorted_segment_sum_backward(
  const THFloatTensor* grad_output_cuda, const THLongTensor* segment_ids_cuda, const int* data_shape, THFloatTensor* grad_data_cuda);

