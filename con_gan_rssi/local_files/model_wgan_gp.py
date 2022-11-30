"""
Discriminator and Generator implementation from DCGAN paper
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_channel, features_d, num_classes, num_aps, input_width):
        super(Discriminator, self).__init__()
        self.num_aps = num_aps
        self.input_width = input_width
        self.input_channel = input_channel
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(self.input_channel+1, features_d, kernel_size=10, stride=1, padding=4),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 10, 1, 4),
            self._block(features_d * 2, features_d * 4, 10, 1, 4),
            self._block(features_d * 4, features_d * 8, 10, 2, 4),
            nn.Conv2d(features_d * 8, 1, kernel_size=12, stride=5, padding=4),
        )
        self.embed = nn.Embedding(num_classes, self.num_aps * input_width)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.num_aps, self.input_width)
        x = torch.cat([x, embedding], dim=1) #N x C x img_size(H) x img_size(W)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self,
                 channels_noise,
                 input_channel,
                 features_g,
                 num_classes,
                 num_aps,
                 input_width,
                 embed_size):
        super(Generator, self).__init__()
        self.num_aps = num_aps
        self.input_channel = input_channel
        self.input_width = input_width
        self.embed_size = embed_size
        self.net = nn.Sequential(
            self._block(channels_noise + embed_size, features_g * 8, 10, 1, 4),  # img: 12x21
            self._block(features_g * 8, features_g * 4, 10, 1, 4),  # img: 13x22
            self._block(features_g * 4, features_g * 2, 10, 1, 4),  # img: 14x23
            self._block(features_g * 2, features_g, 10, 1, 4),  # img: 15x24
            nn.ConvTranspose2d(features_g, input_channel, kernel_size=9, stride=1, padding=6),
            # nn.Tanh(),
            nn.ReLU()
        )
        self.embed = nn.Embedding(num_classes, embed_size * num_aps * input_width)


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        # latent vector z: N x noise_dim x 1 x 1
        embedding = self.embed(labels).view(labels.shape[0], self.embed_size, self.num_aps, self.input_width)
        x = torch.cat([x, embedding], dim=1)  # N x C x img_size(H) x img_size(W)
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"


# test()