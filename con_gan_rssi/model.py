"""
Discriminator and Generator implementation from DCGAN paper
"""

import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_channel, features_d, num_classes, num_aps, input_width):
        super(Discriminator, self).__init__()
        self.num_aps = num_aps
        self.input_width = input_width
        self.input_channel = input_channel
        self.disc = nn.Sequential(
            # input: N x APs x 20 x 11
            nn.Conv2d(self.input_channel+1, features_d, kernel_size=10, stride=1, padding=4),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 10, 1, 4),
            self._block(features_d * 2, features_d * 4, 10, 1, 4),
            self._block(features_d * 4, features_d * 8, 10, 2, 4),
            nn.Conv2d(features_d * 8, 1, kernel_size=12, stride=5, padding=4),
            # if WGAN no sigmoid
            nn.Sigmoid(),
        )
        self.embed = nn.Embedding(num_classes, self.num_aps * input_width)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1,  self.num_aps, self.input_width)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self,
                 channels_noise,
                 input_channel,
                 features_g,
                 num_classes,
                 num_aps,
                 input_width,
                 embed_size
                 ):
        super(Generator, self).__init__()
        self.num_aps = num_aps
        self.input_channel = input_channel
        self.input_width = input_width
        self.embed_size = embed_size
        self.gen = nn.Sequential(
            self._block(channels_noise+embed_size, features_g * 8, 10, 1, 4),  # img: 12x21
            self._block(features_g * 8, features_g * 4, 10, 1, 4),  # img: 13x22
            self._block(features_g * 4, features_g * 2, 10, 1, 4),  # img: 14x23
            self._block(features_g * 2, features_g, 10, 1, 4),  # img: 15x24
            nn.ConvTranspose2d(features_g, input_channel, kernel_size=9, stride=1, padding=6), # img: 11x20
            # nn.Linear(input_channel, input_channel),
            # nn.Tanh(),
            # nn.ReLU(),
        )

        self.embed = nn.Embedding(num_classes, embed_size * num_aps * input_width)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Tanh(),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], self.embed_size, self.num_aps, self.input_width)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N,  num_aps, t = 5, 11, 20
    in_channels = 1
    noise_dim = 100
    labels = torch.full((1,5), 6)
    labels = labels.squeeze(0)
    x = torch.randn((N, in_channels, num_aps, t))
    disc = Discriminator(in_channels, 32, 9, num_aps, t)
    disc_out = disc(x, labels)
    assert disc(x, labels).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 32, 9, num_aps, t, 100)
    z = torch.randn((N, noise_dim, num_aps, t))
    gen_out = gen(z, labels)
    assert gen(z, labels).shape == (N, 1, in_channels, t), "Generator test failed"

# test()

N,num_aps, t = 5, 11, 20
in_channels = 1
noise_dim = 11
labels = torch.full((1,5), 6)
labels = labels.squeeze(0)
x = torch.randn((N, in_channels+1, num_aps, t))
m = nn.Conv2d(2, 32, kernel_size=10, stride=1, padding=4)
output1 = m(x)
m = nn.Conv2d(32, 64, kernel_size=10, stride=1, padding=4)
output2 = m(output1)
m = nn.Conv2d(64, 128, kernel_size=10, stride=1, padding=4)
output3 = m(output2)
m = nn.Conv2d(128, 256, kernel_size=10, stride=2, padding=4)
output4 = m(output3)
m = nn.Conv2d(256, 1, kernel_size=12, stride=5, padding=4)
output5 = m(output4)
output6 = nn.Sigmoid()(output5)

# ap8
m = nn.Conv2d(256, 1, kernel_size=14, stride=7, padding=6)
output5 = m(output4)

# ap6
m = nn.Conv2d(256, 1, kernel_size=16, stride=9, padding=8)
output5 = m(output4)

z = torch.randn((N, 200, 11, 20))
m = nn.ConvTranspose2d(100+100, 256,  kernel_size=10, stride=1, padding=4)
_output1 = m(z)
m = nn.ConvTranspose2d(256, 128, kernel_size=10, stride=1, padding=4)
_output2 = m(_output1)
m = nn.ConvTranspose2d(128, 64,  kernel_size=10, stride=1, padding=4)
_output3 = m(_output2)
m = nn.ConvTranspose2d(64, 32,  kernel_size=10, stride=1, padding=4)
_output4 = m(_output3)
m = nn.ConvTranspose2d(32, 1, kernel_size=9, stride=1, padding=6)
_output5 = m(_output4)

#mnist
# x = torch.randn((N, in_channels+1, 64, 64))
# m = nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1)
# output1 = m(x)
# m = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
# output2 = m(output1)
# m = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
# output3 = m(output2)
# m = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
# output4 = m(output3)
# m = nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=0)
# output5 = m(output4)
