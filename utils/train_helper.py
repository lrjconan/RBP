import os
import torch


def snapshot(model, optimizer, config, step, gpus=[0], tag=None):
  model_snapshot = {
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'step': step
  }

  if tag is not None:
    torch.save(model_snapshot,
               os.path.join(config.save_dir,
                            'model_snapshot_{}.pth'.format(tag)))
  else:
    torch.save(model_snapshot,
               os.path.join(config.save_dir,
                            'model_snapshot_{:07d}.pth'.format(step)))


def load_model(model, file_name, optimizer=None):
  model_snapshot = torch.load(file_name)
  model.load_state_dict(model_snapshot['model'])
  if optimizer is not None:
    optimizer.load_state_dict(model_snapshot['optimizer'])
