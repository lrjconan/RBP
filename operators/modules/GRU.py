import torch
import torch.nn as nn
import torch.nn.functional as F
from operators.functions.GRU import FDGRUCellFunction


class FDGRUCell(nn.Module):

  def __init__(self, grad_id):
    super(FDGRUCell, self).__init__()
    self.GRUCellFunction = FDGRUCellFunction(grad_id)

  def forward(self,
              input,
              hidden,
              d_input,
              d_hidden,
              w_ih,
              w_hh,
              b_ih=None,
              b_hh=None):
    return self.GRUCellFunction(input, hidden, d_input, d_hidden, w_ih, w_hh,
                                b_ih, b_hh)
