import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class FDGRUCellFunction(Function):

  def __init__(self, grad_id):
    self.grad_id = grad_id

  def forward(self, input, hidden, v_i, v_h, w_ih, w_hh, b_ih=None, b_hh=None):
    gi = F.linear(input, w_ih, b_ih)  # input * w_ih.T
    gh = F.linear(hidden, w_hh, b_hh)

    d_gi = F.linear(v_i, w_ih)
    d_gh = F.linear(v_h, w_hh)

    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    d_i_r, d_i_i, d_i_n = d_gi.chunk(3, 1)
    d_h_r, d_h_i, d_h_n = d_gh.chunk(3, 1)

    resetgate = F.sigmoid(i_r + h_r)
    inputgate = F.sigmoid(i_i + h_i)

    newgate = F.tanh(i_n + resetgate * h_n)
    # hy = newgate + inputgate * (hidden - newgate)

    d_resetgate_input = resetgate * (1 - resetgate) * d_i_r
    d_resetgate_h = resetgate * (1 - resetgate) * d_h_r

    d_inputgate_input = inputgate * (1 - inputgate) * d_i_i
    d_inputgate_h = inputgate * (1 - inputgate) * d_h_i

    d_newgate_input = (1 - newgate**2) * (d_i_n + d_resetgate_input * h_n)
    d_newgate_h = (1 - newgate**2) * (d_resetgate_h * h_n + resetgate * d_h_n)

    if self.grad_id == 0:  # calc for input
      d_resetgate = d_resetgate_input + d_resetgate_h
      d_inputgate = d_inputgate_input + d_inputgate_h
      d_newgate = (1 - newgate**2) * (d_i_n + d_resetgate * h_n)
      d_hy = d_newgate + d_inputgate * (hidden - newgate) + inputgate * (
          -d_newgate)
    elif self.grad_id == 1:  # calc for hidden
      d_resetgate = d_resetgate_input + d_resetgate_h
      d_inputgate = d_inputgate_input + d_inputgate_h
      d_newgate = (1 - newgate**2) * (d_resetgate * h_n + resetgate * d_h_n)
      d_hy = d_newgate + d_inputgate * (hidden - newgate) + inputgate * (
          v_h - d_newgate)
    else:  # calc for both
      d_hy_h = inputgate * v_h + d_inputgate_h * (hidden - newgate) + (
          1 - inputgate) * d_newgate_h
      d_hy_input = d_inputgate_input * (hidden - newgate) + (
          1 - inputgate) * d_newgate_input
      d_hy = d_hy_h + d_hy_input

    return d_hy

  def backward(self, grad_output):
    raise NotImplementedError()
