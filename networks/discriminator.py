__author__ = 'JosueCom'
__date__ = '4/21/2020'
__email__ = "josue.n.rivera@outlook.com"

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf

        self.pre = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            nn.AdaptiveMaxPool2d((10,10)))

        self.to_one = nn.Sequential(
            nn.Linear(10 * 10, 30),
            nn.Linear(30, 1)
        )

    def forward(self, img1, img2):
        img1 = self.pre(img1)
        img2 = self.pre(img2)
        img = self.model(img1 + img2)
        return self.to_one(img.view(-1, 10 * 10))

if __name__ == '__main__':
    dis = Discriminator(0, 3, 10)








