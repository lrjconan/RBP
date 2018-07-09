import torch
import torch.nn as nn
import torch.nn.functional as F


class CitationBaseline(nn.Module):

  def __init__(self, config):
    super(CitationBaseline, self).__init__()
    self.config = config
    self.feat_dim = config.model.feat_dim
    self.output_dim = config.model.output_dim

    # output function
    self.output_func = nn.Linear(self.feat_dim, self.output_dim)
    self.loss_func = nn.CrossEntropyLoss()

    self._initialize()

  def _initialize(self):
    modules = [xx for xx in [self.output_func] if xx is not None]

    for m in modules:
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
    assert feat.shape[2] == self.feat_dim

    edge = edge.squeeze(0)
    feat = feat.squeeze(0)
    mask = mask.squeeze(0)
    target = target.squeeze(0)

    y = self.output_func(feat)

    if target is not None:
      loss = self.loss_func(y[mask, :], target[mask])

    if self.training:
      param = [pp for pp in self.parameters()]
      grad = torch.autograd.grad(loss, param)
    else:
      grad = None

    diff_norm_placeholder = torch.zeros([100, 1])

    return y[mask, :], target[mask], diff_norm_placeholder, loss, grad
