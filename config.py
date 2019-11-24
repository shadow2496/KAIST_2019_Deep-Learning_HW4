from easydict import EasyDict


config = EasyDict()

config.name = ''
config.dataset_dir = './datasets/'
config.tensorboard_dir = './tensorboard/'
config.checkpoint_dir = './checkpoints/'

config.batch_size = 128
config.noise_features = 96
config.num_workers = 4

config.gen_lr = 0.001
config.dis_lr = 0.001
config.betas = (0.5, 0.999)

config.print_step = 100
config.tensorboard_step = 100
config.load_epoch = 0
config.train_epochs = 10
config.is_train = True
config.use_cuda = True
