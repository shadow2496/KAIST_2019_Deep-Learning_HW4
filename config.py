from easydict import EasyDict


config = EasyDict()

config.name = ''
config.dataset_dir = './datasets/'
config.tensorboard_dir = './tensorboard/'
config.checkpoint_dir = './checkpoints/'

config.batch_size = 0
config.num_workers = 0

config.lr = 0
config.momentum = 0
config.weight_decay = 0

config.print_step = 0
config.tensorboard_step = 0
config.checkpoint_step = 0
config.load_iter = 0
config.train_iters = 0
config.is_train = True
config.use_cuda = True
