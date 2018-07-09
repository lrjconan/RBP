import math
import torch
import torch.nn as nn
import numpy as np
from model.rbp import RBP
from utils.model_helper import detach_param_with_grad

__all__ = ['HypergradNet']


class MLP(object):

  def __init__(self, hidden_list, has_bias=True, use_gpu=True):
    self.use_gpu = use_gpu
    self.has_bias = has_bias
    self.hidden = hidden_list
    self.num_layers = len(self.hidden) - 1
    self.non_linear = nn.Tanh()

    # intialize paramters
    self.param = []
    for ii in range(self.num_layers):
      # weight
      weight = torch.zeros(
          self.hidden[ii], self.hidden[ii + 1], requires_grad=True)
      weight = weight.cuda() if use_gpu else weight
      self.param += [weight]

      # bias
      if self.has_bias:
        bias = torch.zeros(self.hidden[ii + 1], requires_grad=True)
        bias = bias.cuda() if use_gpu else bias
        self.param += [bias]

    self.init_parameters()

  def init_parameters(self):
    for ii in range(self.num_layers):
      if self.has_bias:
        weight = self.param[ii * 2]
        bias = self.param[ii * 2 + 1]
        # Note: this will cut off the gradient
        nn.init.normal_(weight, 0, 1)
        nn.init.normal_(bias, 0, 1)
        self.param[ii * 2] = weight * np.exp(-3.0)
        self.param[ii * 2 + 1] = bias * np.exp(-3.0)
      else:
        weight = self.param[ii]
        nn.init.normal_(weight, 0.0, 1.0)
        self.param[ii] = weight * np.exp(-3.0)

  def forward(self, data):
    x = data
    for ii in range(self.num_layers):
      if self.has_bias:
        if ii == self.num_layers - 1:
          x = torch.mm(x, self.param[ii * 2]) + self.param[ii * 2 + 1].view(
              1, -1)
        else:
          x = self.non_linear(
              torch.mm(x, self.param[ii * 2]) + self.param[ii * 2 + 1].view(
                  1, -1))
      else:
        if ii == self.num_layers - 1:
          x = torch.mm(x, self.param[ii])
        else:
          x = self.non_linear(torch.mm(x, self.param[ii]))

    return x


class HypergradNet(nn.Module):

  def __init__(self, config):
    super(HypergradNet, self).__init__()
    self.config = config
    self.use_gpu = config.use_gpu
    self.mlp_hidden = config.model.mlp_hidden
    self.truncate_iter = config.model.truncate_iter
    self.batch_size = config.model.batch_size
    self.grad_method = config.model.grad_method
    self.num_sgd_step = config.model.num_sgd_step
    self.wd = float(np.exp(-100))
    assert self.grad_method in ['BPTT', 'TBPTT', 'RBP',
                                'Neumann_RBP'], 'not implemented'

    self.loss = nn.CrossEntropyLoss()

    # create hyperparameters
    self.num_layers = len(config.model.mlp_hidden) - 1
    self.num_params = self.num_layers * 2
    self.register_parameter('hyper_lr',
                            nn.Parameter(
                                torch.ones(self.num_params) * config.model.lr))
    self.register_parameter(
        'hyper_momentum',
        nn.Parameter(torch.ones(self.num_params) * config.model.momentum))

    self.hyper_param = [self.hyper_lr, self.hyper_momentum]

  def forward(self, train_data, train_label, val_data, val_label, meta_step):
    # update function
    def _update(model, momentum, data, label):
      weight = torch.cat([pp.view([-1]) for pp in model.param])
      output = model.forward(data)
      loss = self.loss(output, label)
      grad = torch.autograd.grad(
          loss + self.wd * torch.sum(weight * weight),
          model.param,
          retain_graph=True,
          create_graph=True,
          allow_unused=True)
      return self.sgd_momentum(model.param, momentum, grad) + tuple([loss])

    # intialize model
    torch.manual_seed(self.config.seed)
    torch.cuda.manual_seed_all(self.config.seed)    
    num_train_data = train_data.shape[0]
    train_loss = []

    # create model
    self.model = MLP(self.mlp_hidden, use_gpu=self.use_gpu)

    # set up variables for SGD
    self.alpha = torch.exp(self.hyper_lr)
    self.beta = 1.0 / (1.0 + torch.exp(-self.hyper_momentum))
    momentum = [
        torch.zeros_like(pp, requires_grad=True) for pp in self.model.param
    ]
    print('learning rate & momentum = {} & {}'.format(
        self.alpha.data.cpu().numpy(), self.beta.data.cpu().numpy()))

    # training loop
    num_step_per_epoch = math.ceil(num_train_data / float(self.batch_size))    
    param_2nd_last_dummy, momentum_2nd_last_dummy = None, None
    train_data, train_label = torch.from_numpy(train_data), torch.from_numpy(
        train_label)

    if self.use_gpu:
      train_data, train_label = train_data.cuda(), train_label.cuda()

    for ii in range(self.num_sgd_step):
      # re-shuffle data
      if ii % num_step_per_epoch == 0:
        idx = torch.randperm(num_train_data)
        idx = idx.cuda() if self.use_gpu else idx

      start_idx = (ii * self.batch_size) % num_train_data
      end_idx = min(num_train_data, start_idx + self.batch_size)
      data = train_data[idx[start_idx:end_idx], :]
      label = train_label[idx[start_idx:end_idx]]

      # update
      if 'RBP' in self.grad_method and ii == self.num_sgd_step - 1:
        self.model.param, momentum, loss = _update(self.model, momentum, train_data, train_label)
      else:
        self.model.param, momentum, loss = _update(self.model, momentum, data, label)

      if 'RBP' in self.grad_method and ii == self.num_sgd_step - 2:
        param_2nd_last_dummy = detach_param_with_grad(self.model.param)
        momentum_2nd_last_dummy = detach_param_with_grad(momentum)        
        self.model.param = param_2nd_last_dummy
        momentum = momentum_2nd_last_dummy

      if self.grad_method == 'TBPTT' and ii + 1 + self.truncate_iter == self.num_sgd_step:
        self.model.param = detach_param_with_grad(self.model.param)
        momentum = detach_param_with_grad(momentum)

      train_loss.append(np.log10(loss.data.cpu().numpy()))
      if (ii % 50) == 0:
        print('train loss @ step {} = {}'.format(ii + 1, train_loss[-1]))

    # Note: we could also use validation loss to tune hyperparameters
    train_output = self.model.forward(train_data)
    meta_train_loss = self.loss(train_output, train_label)

    if 'RBP' in self.grad_method:
      grad_param = torch.autograd.grad(
          meta_train_loss,
          self.model.param,
          retain_graph=True,
          allow_unused=True)
      grad_param += tuple([torch.zeros_like(pp) for pp in momentum])
      grad_hyper = RBP(
          self.hyper_param,
          self.model.param + momentum,
          param_2nd_last_dummy + momentum_2nd_last_dummy,
          grad_param,
          truncate_iter=self.truncate_iter,
          rbp_method=self.grad_method)
    else:
      grad_hyper = torch.autograd.grad(meta_train_loss, self.hyper_param)

    return train_loss, np.log10(meta_train_loss.data.cpu().numpy()), grad_hyper

  def sgd_momentum(self, param, momentum, grad):
    """ We implement the same sgd momentum as Maclaurin et al. 2015 """
    new_param = [None] * len(param)
    new_momentum = [None] * len(momentum)

    for ii in range(len(param)):
      # hyper parameters per parameter
      alpha = self.alpha[ii]
      beta = self.beta[ii]

      new_momentum[ii] = beta * momentum[ii] - (1.0 - beta) * grad[ii]
      new_param[ii] = param[ii] + alpha * (beta * momentum[ii] -
                                           (1.0 - beta) * grad[ii])

    return new_param, new_momentum
