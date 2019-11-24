import os

import torch


def load_checkpoints(models, checkpoint_dir, name, load_epoch):
    print("Loading {}...".format(name))
    gen_path = os.path.join(checkpoint_dir, name, '{:02d}-gen.ckpt'.format(load_epoch))
    models.gen.load_state_dict(torch.load(gen_path, map_location=lambda storage, loc: storage))
