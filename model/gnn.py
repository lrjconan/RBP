import numpy as np
import torch
import torch.nn as nn
from operators.functions.unsorted_segment_sum import UnsortedSegmentSumFunction
from operators.modules.GRU import FDGRUCell
from model.rbp import RBP
from utils.model_helper import detach_param_with_grad

EPS = float(np.finfo(np.float32).eps)
unsorted_segment_sum = UnsortedSegmentSumFunction.apply


class GNN(nn.Module):

  def __init__(self, config):
    """ A simplified implementation of GNN for citation networks
    
      Note: for non-GRU update function, CG_RBP does not work since we do not provide forward
            the corresponding auto-diff implementation
    """
    super(GNN, self).__init__()
    self.config = config
    self.hidden_dim = config.model.hidden_dim
    self.output_dim = config.model.output_dim
    self.num_layer = config.model.num_layer
    self.num_prop = config.model.num_prop
    self.truncate_iter = config.model.truncate_iter
    self.num_edge_type = config.model.num_edge_type
    self.aggregate_type = config.model.aggregate_type
    self.grad_method = config.model.grad_method
    assert self.num_layer == 1, "not implemented"
    assert self.num_edge_type == 1, "not implemented"
    assert self.aggregate_type in ['avg', 'sum'], 'not implemented'

    # update function
    if config.model.update_func == 'RNN':
      self.update_func = nn.RNNCell(
          input_size=self.hidden_dim,
          hidden_size=self.hidden_dim,
          nonlinearity='relu')
    elif config.model.update_func == 'GRU':
      self.update_func = nn.GRUCell(
          input_size=self.hidden_dim, hidden_size=self.hidden_dim)
      self.update_func_diff = FDGRUCell(grad_id=2)
    elif config.model.update_func == 'MLP':
      self.update_func = nn.Sequential(
          *[nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()])

    # output function
    if config.model.output_func == 'MLP':
      self.output_func = nn.Sequential(
          *[nn.Linear(self.hidden_dim, self.output_dim)])
    else:
      self.output_func = None

    self.loss_func = nn.CrossEntropyLoss()

    self._initialize()

  def _initialize(self):
    mlp_modules = [xx for xx in [self.output_func] if xx is not None]

    for m in mlp_modules:
      if isinstance(m, nn.Sequential):
        for mm in m:
          if isinstance(mm, nn.Linear):
            nn.init.kaiming_normal_(mm.weight.data)
            if mm.bias is not None:
              mm.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()

    if self.config.model.update_func in ['GRU', 'RNN']:
      for m in [self.update_func]:
        nn.init.kaiming_normal_(m.weight_hh.data)
        nn.init.kaiming_normal_(m.weight_ih.data)
        if m.bias:
          m.bias_hh.data.zero_()
          m.bias_ih.data.zero_()
    elif self.config.model.update_func == 'MLP':
      for m in self.update_func:
        if isinstance(m, nn.Linear):
          nn.init.kaiming_normal_(m.weight.data)
          if m.bias is not None:
            m.bias.data.zero_()

  def forward(self, edge, feat, target=None, mask=None):
    """
      feat: shape |V| X D
      edge: shape |E| X 2
      target: shape |V| X 1
    """
    assert edge.shape[0] == 1, 'batch size must be 1'
    assert feat.shape[2] == self.hidden_dim

    edge, feat, mask, target = edge.squeeze(0), feat.squeeze(0), mask.squeeze(0), target.squeeze(0)
    num_node = feat.shape[0]

    state = [None] * (self.num_prop + 1)
    diff_norm = [None] * self.num_prop

    # state shape
    state[-1] = feat
    edge_in = edge[:, 0]
    edge_out = edge[:, 1].contiguous()

    def _prop(state_old):
      # gather message
      msg = torch.index_select(state_old, 0, edge_in)  # shape: L X D

      # aggregate message
      msg = unsorted_segment_sum(msg.unsqueeze(0), edge_out, num_node).squeeze(
          dim=0)  # shape: M X D

      if self.aggregate_type == 'avg':
        const_ones = torch.ones(1, msg.shape[0], 1).float()
        const_ones = const_ones.cuda() if feat.is_cuda else const_ones
        norm_const = unsorted_segment_sum(const_ones, edge_out,
                                          num_node).squeeze(dim=0)
        msg = msg / (norm_const + EPS)  # shape: M X D
      else:
        pass

      # update state
      state_new = self.update_func(msg, state_old)  # GRU update

      return state_new

    def _prop_forward_diff(input_v, state_old):
      # gather message
      msg = torch.index_select(state_old[0], 0, edge_in)  # shape: L X D
      msg_v = torch.index_select(input_v[0], 0, edge_in)  # shape: L X D

      # aggregate message
      msg = unsorted_segment_sum(msg.unsqueeze(0), edge_out, num_node).squeeze(
          dim=0)  # shape: M X D
      msg_v = unsorted_segment_sum(msg_v.unsqueeze(0), edge_out,
                                   num_node).squeeze(dim=0)  # shape: M X D

      if self.aggregate_type == 'avg':
        const_ones = torch.ones(1, msg.shape[0], 1).float()
        const_ones = const_ones.cuda() if feat.is_cuda else const_ones
        norm_const = unsorted_segment_sum(const_ones, edge_out,
                                          num_node).squeeze(dim=0)
        msg = msg / (norm_const + EPS)  # shape: M X D
        msg_v = msg_v / (norm_const + EPS)  # shape: M X D
      else:
        pass

      # update state
      state_new = self.update_func(msg, state_old[0])  # GRU update
      state_new_v = self.update_func_diff(
          msg, state_old[0], msg_v, input_v[0], self.update_func.weight_ih,
          self.update_func.weight_hh, self.update_func.bias_ih,
          self.update_func.bias_hh)  # GRU update

      return [state_new_v]

    # propagation
    for jj in range(self.num_prop):
      state[jj] = _prop(state[jj - 1])
      diff_norm[jj] = torch.norm(state[jj] - state[jj - 1]) / float(state[jj].numel())

      if self.grad_method == 'TBPTT':
        if jj + 1 + self.truncate_iter == self.num_prop:
          state[jj] = state[jj].detach()        
      elif 'RBP' in self.grad_method:
        if jj + 2 == self.num_prop:
          state[jj] = detach_param_with_grad([state[jj]])[0]        

    # output
    state_last = state[-2]
    state_2nd_last = state[-3]    
    if self.output_func:
      y = self.output_func(state_last)
    else:
      y = state_last

    if target is not None:
      loss = self.loss_func(y[mask, :], target[mask])
    else:
      loss = None

    if self.training:
      if 'RBP' in self.grad_method:
        grad_dict = {}
        tuple_output = [(nn, pp) for nn, pp in self.named_parameters()
                        if 'output_func' in nn]      
        tuple_update = [(nn, pp) for nn, pp in self.named_parameters()
                        if 'update_func' in nn]
        name_output, param_output = zip(*tuple_output)
        name_update, param_update = zip(*tuple_update)
        grad_output = torch.autograd.grad(loss, param_output, retain_graph=True)
        for nn, gg in zip(name_output, grad_output):
          grad_dict[nn] = gg

        grad_state_last = torch.autograd.grad(loss, state_last, retain_graph=True)        
        grad_update = RBP(param_update,
                [state_last],
                [state_2nd_last],
                grad_state_last,
                update_forward_diff=_prop_forward_diff,
                eta=1.0e-5,
                truncate_iter=self.truncate_iter,
                rbp_method=self.grad_method)
        for nn, gg in zip(name_update, grad_update):
          grad_dict[nn] = gg

        grad = [grad_dict[nn] for nn, pp in self.named_parameters()]      
      else:
        param = [pp for pp in self.parameters()]
        grad = torch.autograd.grad(loss, param)
    else:
      grad = None

    return y[mask, :], target[mask], torch.stack(diff_norm), loss, grad
