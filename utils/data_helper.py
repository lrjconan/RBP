import numpy as np
from sklearn.decomposition import PCA


def read_idx_file(file_name):
  idx = []
  with open(file_name) as f:
    for line in f:
      idx += [int(line)]

  return idx


def preprocess_feature(feature, norm_method=None):
  """ Normalize feature matrix """

  if norm_method == "L1":
    # L1 norm
    feature /= (feature.sum(1, keepdims=True) + EPS)

  elif norm_method == "L2":
    # L2 norm
    feature /= (np.sqrt(np.square(feature).sum(1, keepdims=True)) + EPS)

  elif norm_method == "std":
    # Standardize
    std = np.std(feature, axis=0, keepdims=True)
    feature -= np.mean(feature, 0, keepdims=True)
    feature /= (std + EPS)
  else:
    # nothing
    pass

  return feature


def pca(X, k=16, seed=1234):
  pca = PCA(n_components=k, random_state=seed)
  return pca.fit_transform(X)
