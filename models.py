import torch
from torch import nn


class Maxout(nn.Module):
    def __init__(self, in_features, out_features):
        super(Maxout, self).__init__()
        self.layer1 = nn.Linear(in_features, out_features)
        self.layer2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        output1 = self.layer1(x)
        output2 = self.layer2(x)
        return torch.max(output1, output2)


class Generator(nn.Module):
    def __init__(self, in_features):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x) * 0.5 + 0.5


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            Maxout(784, 256),
            nn.Dropout(),
            Maxout(256, 256),
            nn.Dropout(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.layers(x)


class GAN(nn.Module):
    def __init__(self, config):
        super(GAN, self).__init__()

        self.gen = Generator(config.noise_features)
        if config.is_train:
            self.dis = Discriminator()

    def forward(self, x):
        pass
