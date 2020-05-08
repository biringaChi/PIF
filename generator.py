__author__ = 'biringaChidera'
__email__ = "biringaChidera@gmail.com"

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, nc, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ngf = ngf

        self.preprocess = nn.Sequential(
            nn.Conv2d(nc, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True))

        self.encoder = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(ngf * 16, ngf * ngf, kernel_size=4, stride=1, padding=1, bias=False),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * ngf, ngf * 16, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh())

    def forward(self, img1, img2):
        img = self.preprocess(img1) + self.preprocess(img2)
        img = self.encoder(img)
        img = self.decoder(img)
        return img


if __name__ == '__main__':
    gen = Generator(0, 3, 32)
#   print(gen)
