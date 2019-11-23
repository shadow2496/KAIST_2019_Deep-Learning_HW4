from torch import nn


# class Maxout(nn.Module)


class Generator(nn.Module):
    def __init__(self, in_features, out_features):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_features),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x) * 0.5 + 0.5


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.layers(x)
