import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from operators.functions.unsorted_segment_sum import UnsortedSegmentSumFunction


class FDUnsortedSegmentSum(nn.Module):

  def __init__(self, num_segments):
    super(FDUnsortedSegmentSum, self).__init__()
    self.num_segments = num_segments

  def forward(self, data, v, segment_index, dim=1):

    assert (data.size() == v.size()), "{} vs {}".format(data.size(), v.size())

    if dim != 1:
      v = v.transpose(dim, 1).contiguous()
    size_prev = list(v.size())
    v = v.view(size_prev[0], size_prev[1], -1)
    v = UnsortedSegmentSumFunction.apply(v, segment_index, self.num_segments)
    size_prev[1] = self.num_segments
    v = v.view(size_prev)
    if dim != 1:
      v = v.transpose(dim, 1).contigous()
    return v


class UnsortedSegmentSum(nn.Module):

  def __init__(self, num_segments):
    super(UnsortedSegmentSum, self).__init__()
    self.num_segments = num_segments

  def forward(self, data, segment_index):
    return UnsortedSegmentSumFunction.apply(data, segment_index,
                                            self.num_segments)
