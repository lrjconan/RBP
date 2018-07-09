import os
import pickle
import numpy as np
import torch
from torchvision import datasets, transforms

__all__ = ['load_data_dicts']


class MNIST(datasets.MNIST):

  def __init__(self,
               root,
               train=True,
               transform=None,
               target_transform=None,
               download=True):
    super(MNIST, self).__init__(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download)

    self.train_data, self.train_labels = torch.load(
        os.path.join(self.root, self.processed_folder, self.training_file))
    self.test_data, self.test_labels = torch.load(
        os.path.join(self.root, self.processed_folder, self.test_file))

  def get_data(self):
    return self.train_data.data.cpu().numpy(), self.train_labels.data.cpu(
    ).numpy(), self.test_data.data.cpu().numpy(), self.test_labels.data.cpu(
    ).numpy()


def load_data(normalize=False):
  dataset = MNIST('./data/mnist')
  train_images, train_labels, test_images, test_labels = dataset.get_data()

  one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
  partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
  train_images = partial_flatten(train_images) / 255.0
  test_images = partial_flatten(test_images) / 255.0
  train_labels = one_hot(train_labels, 10)
  test_labels = one_hot(test_labels, 10)
  N_data = train_images.shape[0]

  if normalize:
    train_mean = np.mean(train_images, axis=0)
    train_images = train_images - train_mean
    test_images = test_images - train_mean
  return train_images, train_labels, test_images, test_labels


def load_data_subset(*args):
  train_images, train_labels, test_images, test_labels = load_data(
      normalize=True)
  all_images = np.concatenate((train_images, test_images), axis=0)
  all_labels = np.concatenate((train_labels, test_labels), axis=0)
  datapairs = []
  start = 0
  for N in args:
    end = start + N
    datapairs.append((all_images[start:end], all_labels[start:end]))
    start = end
  return datapairs


def load_data_dicts(*args):
  datapairs = load_data_subset(*args)
  return [{"X": dat[0], "T": dat[1]} for dat in datapairs]
