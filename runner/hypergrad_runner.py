from __future__ import (division, print_function)
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from model import *
from dataset import *

from utils.arg_helper import mkdir
from utils.logger import get_logger
from utils.train_helper import snapshot, load_model

logger = get_logger('exp_logger')

__all__ = ['HypergradRunner']


class HypergradRunner(object):

  def __init__(self, config):
    self.config = config
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus    
    mkdir(self.dataset_conf.path)

  def train(self):
    train_data_dict, val_data_dict, test_data_dict = load_data_dicts(
        10000, 50000, 10000)

    train_data = train_data_dict['X'].astype(np.float32)
    val_data = val_data_dict['X'].astype(np.float32)
    test_data = test_data_dict['X'].astype(np.float32)
    train_label = np.argmax(train_data_dict['T'], axis=1)
    val_label = np.argmax(val_data_dict['T'], axis=1)
    test_label = np.argmax(test_data_dict['T'], axis=1)

    # create models
    model = eval(self.model_conf.name)(self.config)

    # create optimizer
    params = model.hyper_param
    if self.train_conf.meta_optimizer == 'SGD':
      optimizer = optim.SGD(
          params,
          lr=self.train_conf.meta_lr,
          momentum=self.train_conf.meta_momentum,
          weight_decay=0.0)
    elif self.train_conf.meta_optimizer == 'Adam':
      optimizer = optim.Adam(
          params, lr=self.train_conf.meta_lr, weight_decay=0.0)
    else:
      raise ValueError("Non-supported meta optimizer!")

    # reset gradient
    optimizer.zero_grad()

    # resume training
    if self.train_conf.is_resume:
      load_model(model, self.train_conf.resume_model, optimizer=optimizer)

    if self.use_gpu:
      model = nn.DataParallel(model, device_ids=self.gpus).cuda()

    # Training Loop
    best_meta_train_loss = 10.0
    results = defaultdict(list)
    for ii in range(self.train_conf.max_meta_iter):
      optimizer.zero_grad()
      train_loss, meta_train_loss, grad_hyper = model(train_data, train_label,
                                               val_data, val_label, ii)

      # decay meta learning rate
      if ii == 50:
        for pg in optimizer.param_groups:
          pg['lr'] *= 0.1

      # meta optimization step
      for p, g in zip(params, grad_hyper):
        p.grad = g

      optimizer.step()
      print('Meta validation loss @step {} = {}'.format(ii + 1, meta_train_loss))
      if meta_train_loss < best_meta_train_loss:
        best_meta_train_loss = meta_train_loss

      results['meta_train_loss'] += [meta_train_loss]
      results['last_train_loss'] = train_loss

    pickle.dump(results,
                open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    return best_meta_train_loss