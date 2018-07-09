from __future__ import (division, print_function)
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import *
from dataset import *
from collections import defaultdict

from utils.arg_helper import mkdir
from utils.logger import get_logger
from utils.train_helper import snapshot, load_model

logger = get_logger('exp_logger')

__all__ = ['HopfieldRunner']


def binarize_data(data):
  data[data > 0.0] = 1.0
  data[data <= 0.0] = 0.0


class BinaryMNIST(datasets.MNIST):

  def __init__(self,
               root,
               num_imgs=1,
               train=True,
               transform=None,
               target_transform=None,
               download=False):
    super(BinaryMNIST, self).__init__(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download)

    self.train_data, self.train_labels = torch.load(
        os.path.join(self.root, self.processed_folder, self.training_file))
    self.test_data, self.test_labels = torch.load(
        os.path.join(self.root, self.processed_folder, self.test_file))

    self.num_imgs = num_imgs
    idx_list = []
    for ii in range(self.num_imgs):
      idx_list += [torch.nonzero(self.train_labels == ii)[0]]

    self.train_data = torch.cat([self.train_data[ii] for ii in idx_list], dim=0)

    self.train_data = self.train_data[:self.num_imgs].float() / 255.0
    for ii in range(self.num_imgs):
      binarize_data(self.train_data)

    self.train_data = (self.train_data * 255.0).byte()
    self.test_data = self.train_data.clone()

  def __len__(self):
    return self.num_imgs


class HopfieldRunner(object):

  def __init__(self, config):
    self.config = config
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train    
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    mkdir(self.dataset_conf.path)

  def train(self):
    # create data loader
    train_dataset = BinaryMNIST(
        self.dataset_conf.path,
        num_imgs=self.dataset_conf.num_imgs,
        train=True,
        transform=transforms.ToTensor(),
        download=True)
    val_dataset = BinaryMNIST(
        self.dataset_conf.path,
        num_imgs=self.dataset_conf.num_imgs,
        train=False,
        transform=transforms.ToTensor(),
        download=True)

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
    results = defaultdict(list)
    for epoch in range(self.train_conf.max_epoch):
      # validation
      if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
        model.eval()
        val_loss = []
        val_counter = 0
        for imgs, labels in val_loader:
          if self.use_gpu:
            imgs, labels = imgs.cuda(), labels.cuda()

          imgs, labels = imgs.float(), labels.float()
          imgs_corrupt = self.rand_corrupt(
              imgs, corrupt_level=self.dataset_conf.corrupt_level)
          curr_loss, imgs_memory, _, _ = model(imgs_corrupt)
          img_recover = imgs_memory[-self.model_conf.input_dim:]
          img_recover_show = img_recover.clone().detach()
          img_recover_show.requires_grad = False
          img_recover_show[img_recover_show >= 0.5] = 1.0
          img_recover_show[img_recover_show < 0.5] = 0.0
          val_loss += [float(curr_loss.data.cpu().numpy())]
          val_counter += 1

        val_loss = float(np.mean(val_loss))
        logger.info("Avg. Validation Loss = {}".format(np.log10(val_loss)))
        results['val_loss'] += [val_loss]
        model.train()

      # training
      lr_scheduler.step()
      for imgs, labels in train_loader:
        if self.use_gpu:
          imgs, labels = imgs.cuda(), labels.cuda()

        imgs, labels = imgs.float(), labels.float()
        optimizer.zero_grad()
        train_loss, imgs_memory, diff_norm, grad = model(imgs)

        for pp, ww in zip(model.parameters(), grad):
          pp.grad = ww

        optimizer.step()
        train_loss = float(train_loss.data.cpu().numpy())
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]

        # display loss
        if iter_count % self.train_conf.display_iter == 0:
          logger.info("Loss @ epoch {:04d} iteration {:08d} = {}".format(
              epoch + 1, iter_count + 1, np.log10(train_loss)))

          tmp_key = 'diff_norm_{}'.format(iter_count + 1)
          results[tmp_key] = diff_norm

        iter_count += 1

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module
                 if self.use_gpu else model, optimizer, self.config, epoch + 1)

    pickle.dump(results,
                open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))

  def rand_corrupt(self, img, corrupt_level=0.1):

    original_shape = img.shape
    img = img.view([28 * 28])
    idx = torch.nonzero(img).squeeze()
    img_corrupt = img.clone()
    corrupt_num = int(len(idx) * corrupt_level)
    npr = np.random.RandomState(self.config.seed)
    idx_perm = npr.permutation(len(idx))
    img_corrupt[idx[idx_perm[:corrupt_num]]] = 0.0
    img_corrupt = img_corrupt.view(original_shape)

    return img_corrupt
