from __future__ import (division, print_function)
import os
import pickle
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import *
from dataset import *
from utils.arg_helper import mkdir
from utils.logger import get_logger
from utils.train_helper import snapshot, load_model

logger = get_logger('exp_logger')

__all__ = ['CitationRunner']


class CitationRunner(object):

  def __init__(self, config):
    self.config = config
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    mkdir(self.dataset_conf.path)
    assert self.dataset_conf.name in ['cora',
                                      'pubmed'], "Non-supported datasets!"

  def train(self):
    # create data loader
    train_dataset = Citation(
        self.dataset_conf.path,
        feat_dim_pca=self.model_conf.feat_dim,
        dataset_name=self.dataset_conf.name,
        split='train',
        train_ratio=self.dataset_conf.train_ratio,
        use_rand_split=self.dataset_conf.rand_split,
        seed=self.config.seed)
    val_dataset = Citation(
        self.dataset_conf.path,
        feat_dim_pca=self.model_conf.feat_dim,
        dataset_name=self.dataset_conf.name,
        split='val',
        train_ratio=self.dataset_conf.train_ratio,
        use_rand_split=self.dataset_conf.rand_split,
        seed=self.config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=self.train_conf.shuffle,
        num_workers=self.train_conf.num_workers,
        drop_last=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=False,
        num_workers=self.train_conf.num_workers,
        drop_last=False)

    # create models
    model = eval(self.model_conf.name)(self.config)

    # create optimizer
    params = model.parameters()
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
          params,
          lr=self.train_conf.lr,
          momentum=self.train_conf.momentum,
          weight_decay=self.train_conf.wd)
    elif self.train_conf.optimizer == 'Adam':
      optimizer = optim.Adam(
          params, lr=self.train_conf.lr, weight_decay=self.train_conf.wd)
    else:
      raise ValueError("Non-supported optimizer!")

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=self.train_conf.lr_decay_steps,
        gamma=self.train_conf.lr_decay)

    # reset gradient
    optimizer.zero_grad()

    # resume training
    if self.train_conf.is_resume:
      load_model(model, self.train_conf.resume_model, optimizer=optimizer)

    if self.use_gpu:
      model = nn.DataParallel(model, device_ids=self.gpus).cuda()

    # Training Loop
    iter_count = 0
    best_val_acc = .0
    results = defaultdict(list)
    for epoch in range(self.train_conf.max_epoch):
      # validation
      if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
        model.eval()
        val_loss = []
        total, correct = .0, .0
        for node_feat, node_label, edge, mask in val_loader:
          if self.use_gpu:
            node_feat, node_label, edge, mask = node_feat.cuda(
            ), node_label.cuda(), edge.cuda(), mask.cuda()

          node_feat, node_label, edge, mask = node_feat.float(
          ), node_label.long(), edge.long(), mask.byte()

          node_logit, node_label, _, curr_loss, _ = model(
              edge, node_feat, target=node_label, mask=mask)
          val_loss += [float(curr_loss.data.cpu().numpy())]
          _, predicted = torch.max(node_logit.data, 1)
          total += node_label.size(0)
          correct += predicted.eq(node_label.data).cpu().numpy().sum()

        val_loss = float(np.mean(val_loss))
        val_acc = 100.0 * correct / total

        # save best model
        if val_acc > best_val_acc:
          best_val_acc = val_acc
          snapshot(model, optimizer, self.config, epoch + 1, tag='best')

        logger.info("Avg. Validation Loss = {}".format(val_loss))
        logger.info("Validation Accuracy = {}".format(val_acc))
        logger.info(
            "Current Best Validation Accuracy = {}".format(best_val_acc))
        results['val_loss'] += [val_loss]
        results['val_acc'] += [val_acc]
        model.train()

      # training
      lr_scheduler.step()
      for node_feat, node_label, edge, mask in train_loader:
        if self.use_gpu:
          node_feat, node_label, edge, mask = node_feat.cuda(), node_label.cuda(
          ), edge.cuda(), mask.cuda()

        node_feat, node_label, edge, mask = node_feat.float(), node_label.long(
        ), edge.long(), mask.byte()
        # optimizer.zero_grad()

        node_logit, _, diff_norm, train_loss, grad_w = model(
            edge, node_feat, target=node_label, mask=mask)

        # assign gradient
        for pp, ww in zip(model.parameters(), grad_w):
          pp.grad = ww

        optimizer.step()
        train_loss = float(train_loss.data.cpu().numpy())
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]

        # display loss
        if (iter_count + 1) % self.train_conf.display_iter == 0:
          logger.info("Loss @ epoch {:04d} iteration {:08d} = {}".format(
              epoch + 1, iter_count + 1, train_loss))
          tmp_key = 'diff_norm_{}'.format(iter_count + 1)
          results[tmp_key] = diff_norm.data.cpu().numpy().tolist()

        iter_count += 1

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module
                 if self.use_gpu else model, optimizer, self.config, epoch + 1)

    results['best_val_acc'] += [best_val_acc]
    pickle.dump(results,
                open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))

    return best_val_acc
