__author__ = 'biringaChidera'

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nc, ndf):
        super(Generator, self).__init__()
        self.nc = nc
        self.ndf = ndf

        self.preprocess = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(True))

        self.encoder = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(True),
            nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(True),
            nn.Conv2d(ndf * 16, ndf * ndf, kernel_size=4, stride=1, bias=False), nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ndf * ndf, ndf * 16, kernel_size=4, stride=1, bias=False), nn.BatchNorm2d(ndf * ndf), nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 16, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ndf * 16), nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ndf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ndf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ndf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ndf, nc, kernel_size=4, stride=2, padding=1, bias=False), nn.Tanh())

    def forward(self, img1, img2):
        img = self.preprocess(img1) + self.preprocess(img2)
        img = self.encoder(img)
        img = self.decoder(img)
        return img

if __name__ == '__main__':
    gen = Generator()
    print(gen)








