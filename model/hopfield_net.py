import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rbp import RBP
from utils.model_helper import detach_param_with_grad


class HopfieldNet(nn.Module):

  def __init__(self, config):
    """ Hopfield Networks

        config must contain following parameters:
          input_dim: dimension of node input
          hidden_dim: dimension of hidden state, typically 512
          output_dim: same as input_dim for reconstructing images
          num_update: # of update
          truncate_iter: # of truncation step in Neumann RBP
          grad_method: ['BPTT', 'TBPTT', 'RBP', 'CG_RBP', 'Neumann_RBP']
    """
    super(HopfieldNet, self).__init__()
    self.input_dim = config.model.input_dim
    self.hidden_dim = config.model.hidden_dim
    self.output_dim = self.input_dim
    self.all_dim = self.input_dim + self.hidden_dim + self.output_dim
    self.num_update = config.model.num_update
    self.grad_method = config.model.grad_method
    self.truncate_iter = config.model.truncate_iter
    self.loss_func = nn.L1Loss()

    self.W = nn.Parameter(torch.randn(self.all_dim, self.all_dim) * 1.0e-3)
    self.non_linear = nn.Sigmoid()

  def forward(self, feat):
    assert feat.shape[0] == 1, "set batch size to 1 for simplicity"
    feat = feat.view([-1, 1]).float()
    pad_array = (0, 0, 0, self.hidden_dim + self.output_dim)
    feat = F.pad(feat, pad_array)
    mask = torch.ones_like(feat)
    mask[:self.input_dim] = 0.0
    diag_mask = torch.ones(
        [self.all_dim, self.all_dim]) - torch.eye(self.all_dim)
    diag_mask = diag_mask.cuda() if feat.is_cuda else diag_mask
    W = (self.W + self.W.transpose(0, 1)) * diag_mask  # remove self loop
    # W = (self.W + self.W.transpose(0, 1))

    a = 1.0
    b = 0.5
    state = [None] * (self.num_update + 1)
    state[-1] = feat
    diff_norm = [None] * self.num_update

    def _update(state_old):      
      state_new = (1 - a) * state_old + torch.mm(
          W, self.non_linear(state_old * b))
      return state_new * mask + feat * (1 - mask)

    def _update_forward_diff(input_v, state_old):
      # input_v, state_old must be list although there is one element
      sig_wv = self.non_linear(torch.mm(W, state_old[0]))
      state_wv = sig_wv * (1 - sig_wv) * torch.mm(W, input_v[0])
      state_v = (1 - a) * input_v[0] + b * state_wv
      state_v = state_v * mask
      return [state_v]

    # update
    for jj in range(self.num_update):
      state[jj] = _update(state[jj - 1])
      diff_norm[jj] = torch.norm(state[jj] - state[jj - 1]).data.cpu().numpy()

      if self.grad_method == 'TBPTT':
        if jj + 1 + self.truncate_iter == self.num_update:
          state[jj] = state[jj].detach()
      elif 'RBP' in self.grad_method:
        if jj + 2 == self.num_update:
          state[jj] = detach_param_with_grad([state[jj]])[0]

    state_last = state[-2]
    state_2nd_last = state[-3]

    if self.training:
      loss = self.loss_func(state_last[-self.output_dim:], feat[:self.input_dim])
      grad_state_last = torch.autograd.grad(loss, state_last, retain_graph=True)
      params = [pp for pp in self.parameters()]

      if 'RBP' in self.grad_method:
        grad = RBP(params,
                   [state_last],
                   [state_2nd_last],
                   grad_state_last,
                   update_forward_diff=_update_forward_diff,
                   truncate_iter=self.truncate_iter,
                   rbp_method=self.grad_method)
      else:
        params = [pp for pp in self.parameters()]
        grad = torch.autograd.grad(loss, params)
    else:
      binary_state = state_last[-self.output_dim:]
      binary_state[binary_state >= 0.5] = 1.0
      binary_state[binary_state < 0.5] = 0.0
      loss = self.loss_func(binary_state, feat[:self.input_dim])
      grad = None

    return loss, state_last, diff_norm, grad
