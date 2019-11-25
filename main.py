import os

import numpy as np
from sklearn.neighbors.kde import KernelDensity
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from config import config
from models import GAN
from utils import load_checkpoints


def train(models, writer, device):
    train_dataset = datasets.MNIST(config.dataset_dir, train=True, transform=transforms.ToTensor(), download=True)
    val_dataset = datasets.MNIST(config.dataset_dir, train=True, transform=transforms.ToTensor())

    indices = list(range(len(train_dataset)))
    train_sampler = SubsetRandomSampler(indices[:55000])
    val_sampler = SubsetRandomSampler(indices[55000:])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              sampler=train_sampler, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            sampler=val_sampler, num_workers=config.num_workers)

    criterion = nn.BCEWithLogitsLoss()
    gen_optimizer = optim.Adam(models.gen.parameters(), lr=config.gen_lr, betas=config.betas)
    dis_optimizer = optim.Adam(models.dis.parameters(), lr=config.dis_lr, betas=config.betas)

    idx = 0
    real_label = torch.ones(1, device=device)
    fake_label = torch.zeros(1, device=device)
    for _ in range(config.train_epochs):
        for data, _ in train_loader:
            data = data.view(-1, 784).to(device)

            # ----------------------------------------------------------------------------------------------------------
            #                                          Train the discriminator
            # ----------------------------------------------------------------------------------------------------------
            noise = torch.rand((data.size(0), config.noise_features), device=device) * 2 - 1
            generated = models.gen(noise)

            data_logits = models.dis(data)
            generated_logits = models.dis(generated.detach())

            real_labels = real_label.expand_as(data_logits)
            fake_labels = fake_label.expand_as(data_logits)
            dis_loss = criterion(data_logits, real_labels) + criterion(generated_logits, fake_labels)

            dis_optimizer.zero_grad()
            dis_loss.backward()
            dis_optimizer.step()

            # ----------------------------------------------------------------------------------------------------------
            #                                            Train the generator
            # ----------------------------------------------------------------------------------------------------------
            noise = torch.rand((data.size(0), config.noise_features), device=device) * 2 - 1
            generated = models.gen(noise)

            generated_logits = models.dis(generated)
            gen_loss = criterion(generated_logits, real_labels)

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            if (idx + 1) % config.print_step == 0:
                print("Iteration: {}/{}, gen_loss: {:.4f}, dis_loss: {:.4f}".format(
                    idx + 1, (55000 // config.batch_size + 1) * config.train_epochs, gen_loss.item(), dis_loss.item()))

            if (idx + 1) % config.tensorboard_step == 0:
                writer.add_scalar('Loss/generator', gen_loss.item(), idx + 1)
                writer.add_scalar('Loss/discriminator', dis_loss.item(), idx + 1)

                data = data.view(-1, 1, 28, 28)
                generated = generated.view(-1, 1, 28, 28)
                writer.add_images('Result/data', data, idx + 1)
                writer.add_images('Result/generated', generated, idx + 1)

            idx += 1

    gen_path = os.path.join(config.checkpoint_dir, config.name, '{:02d}-gen.ckpt'.format(config.train_epochs))
    dis_path = os.path.join(config.checkpoint_dir, config.name, '{:02d}-dis.ckpt'.format(config.train_epochs))
    torch.save(models.gen.state_dict(), gen_path)
    torch.save(models.dis.state_dict(), dis_path)


def test(models, device):
    test_dataset = datasets.MNIST(config.dataset_dir, train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    X_data = np.zeros((10000, 784), dtype=np.float32)
    X_generated = np.zeros((10000, 784), dtype=np.float32)
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            noise = torch.rand((data.size(0), config.noise_features), device=device) * 2 - 1
            generated = models.gen(noise)

            start = i * config.batch_size
            end = min((i + 1) * config.batch_size, 10000)
            X_data[start:end] = data.view(-1, 784).numpy()
            X_generated[start:end] = generated.cpu().numpy()

    print("Calculating the score...")
    kde = KernelDensity(bandwidth=0.2).fit(X_generated)
    print("Score: {:.4f}".format(kde.score(X_data) / 10000))


def main():
    if not os.path.exists(os.path.join(config.tensorboard_dir, config.name)):
        os.makedirs(os.path.join(config.tensorboard_dir, config.name))
    if not os.path.exists(os.path.join(config.checkpoint_dir, config.name)):
        os.makedirs(os.path.join(config.checkpoint_dir, config.name))

    device = torch.device('cuda:0' if config.use_cuda else 'cpu')
    models = GAN(config).to(device)
    if config.load_epoch != 0:
        load_checkpoints(models, config.checkpoint_dir, config.name, config.load_epoch)

    if config.is_train:
        models.train()
        writer = SummaryWriter(log_dir=os.path.join(config.tensorboard_dir, config.name))
        train(models, writer, device)
    else:
        models.eval()
        test(models, device)


if __name__ == '__main__':
    main()
