import os
import pickle
import numpy as np
from functools import reduce
from torch.utils.data import Dataset
from utils.data_helper import preprocess_feature, read_idx_file, pca


class Citation(Dataset):

  def __init__(self,
               root_dir,
               feat_dim_pca=128,
               dataset_name='cora',
               split='train',
               use_rand_split=False,
               train_ratio=0.05,
               seed=1234):
    self._seed = seed
    self._npr = np.random.RandomState(seed)
    self.split = split
    self.train_ratio = train_ratio
    self.feat_dim_pca = feat_dim_pca
    assert self.train_ratio < 0.5, "too many training samples"
    assert split == 'train' or split == 'val' or split == 'test', "not implemented"

    # read data from disk
    suffixes = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index']
    var_names = [
        'train_feat', 'train_label', 'test_feat', 'test_label', 'all_feat',
        'all_label', 'graph', 'test_id'
    ]

    data_dict = {}
    base_name = '.'.join(['ind', dataset_name])

    for nn, ss in zip(var_names, suffixes):
      if nn == 'test_id':
        data_dict[nn] = read_idx_file(
            os.path.join(root_dir, '.'.join([base_name, ss])))
      else:
        data_dict[nn] = pickle.load(
            open(os.path.join(root_dir, '.'.join([base_name, ss])), 'rb'),
            encoding='latin1')

    # get statistics
    self.num_feat = data_dict['all_feat'].shape[0]
    self.num_nodes = len(data_dict['graph'].keys())
    self.num_train = data_dict['train_feat'].shape[0]
    self.num_test = data_dict['test_feat'].shape[0]
    self.feat_dim = data_dict['train_feat'].shape[1]

    feat_all = preprocess_feature(
        np.concatenate(
            [data_dict['all_feat'].toarray(), data_dict['test_feat'].toarray()],
            axis=0),
        norm_method=None)
    label_all = np.argmax(
        np.concatenate(
            [data_dict['all_label'], data_dict['test_label']], axis=0),
        axis=1)
    idx_all = np.concatenate(
        [np.arange(self.num_feat), np.array(data_dict['test_id'])], axis=0)

    # reduce dimension by PCA
    feat_all_pca = pca(feat_all, k=self.feat_dim_pca)

    # get split
    if use_rand_split:
      perm_idx = self._npr.permutation(len(idx_all))
      self.num_train = int(len(idx_all) * self.train_ratio)
      self.num_val = int(len(idx_all) * (0.5 - self.train_ratio))
      self.num_test = len(idx_all) - self.num_train - self.num_val

      split_train_idx = perm_idx[:self.num_train]
      split_val_idx = perm_idx[self.num_train:self.num_train + self.num_val]
      split_test_idx = perm_idx[self.num_train + self.num_val:self.num_train +
                                self.num_val + self.num_test]
    else:
      # using the inductive split as the Planetoid paper
      split_train_idx = np.arange(self.num_train)
      split_val_idx = np.arange(self.num_train, self.num_train + self.num_val)
      split_test_idx = np.arange(self.num_feat, self.num_feat + self.num_test)

    self.train_idx = idx_all[split_train_idx]
    self.val_idx = idx_all[split_val_idx]
    self.test_idx = idx_all[split_test_idx]

    self.edges = np.array(
        reduce(lambda x, y: x + y, [
            list(zip([xx] * len(vv), vv))
            for xx, vv in data_dict['graph'].items()
        ]),
        dtype=np.int32)

    self.node_feat = np.zeros(
        [self.num_nodes, self.feat_dim_pca], dtype=np.float32)
    self.node_label = np.zeros([self.num_nodes], dtype=np.int32)
    self.node_feat[idx_all, :] = feat_all_pca
    self.node_label[idx_all] = label_all

  def __len__(self):
    return 1

  def __getitem__(self, idx):
    mask = np.zeros([self.num_nodes], dtype=np.int32)

    if self.split == 'train':
      mask[self.train_idx] = 1
    elif self.split == 'val':
      mask[self.val_idx] = 1
    else:
      mask[self.test_idx] = 1

    edges = self.edges.copy()
    node_feat = self.node_feat.copy()
    node_label = self.node_label.copy()

    return node_feat, node_label, edges, mask
