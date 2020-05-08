from __future__ import print_function
import torch
import torch.nn as nn
import pickle
import data_prep as prep
from torchvision import transforms
import torch.nn.parallel
import numpy as np
from generator import Generator
from discriminator import Discriminator
import matplotlib.pyplot as plt

__author__ = 'JosueCom'
__date__ = '5/8/2020'
__email__ = "josue.n.rivera@outlook.com"


#image_size = 500
nc = 3
ngf = 64
num_epochs = 5
beta1 = 0.5
ngpu = 1
lf_to_rg_ratio = 0.5

diff_pickle = open("planet_earth_diff.pickle","rb")

dataset = prep.PIFDataset(
    path='data_prepocessing/PlanetEarth',
    diff = pickle.load(diff_pickle),
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]))

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

## generator
netG = Generator(ngpu, nc, ngf).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

samp = dataset[0]
out = netG(samp["prev"].unsqueeze(0).to(device), samp["next"].unsqueeze(0).to(device))

img = np.uint8(out.cpu().detach().numpy())[:, ::-1, :, :]

print(img.shape)

#img = transforms.ToPILImage()(img)

#print(img.size)

plt.imshow(img)
plt.show()



