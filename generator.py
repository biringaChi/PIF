import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=1, bias=False), nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False), nn.Tanh())

    def forward(self, img1, img2):
        img = torch.cat(img1, img2)
        img = self.encoder(img)
        img = self.decoder(img)
        return self.model(img)

if __name__ == '__main__':
    gen = Generator()
    print(gen)

