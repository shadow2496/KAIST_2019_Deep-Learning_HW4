from torch import nn


# class Maxout(nn.Module)


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
            nn.Linear(784, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
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
