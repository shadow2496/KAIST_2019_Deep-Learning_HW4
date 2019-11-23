import os

import torch
from torch.utils.tensorboard import SummaryWriter

from config import config


def main():
    if not os.path.exists(os.path.join(config.tensorboard_dir, config.name)):
        os.makedirs(os.path.join(config.tensorboard_dir, config.name))
    if not os.path.exists(os.path.join(config.checkpoint_dir, config.name)):
        os.makedirs(os.path.join(config.checkpoint_dir, config.name))

    device = torch.device('cuda:0' if config.use_cuda else 'cpu')
    if config.load_iter != 0:
        pass

    if config.is_train:
        writer = SummaryWriter(log_dir=os.path.join(config.tensorboard_dir, config.name))
    else:
        pass


if __name__ == '__main__':
    main()
