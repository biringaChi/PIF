from __future__ import print_function
import torch
import torch.nn as nn
import pickle
import data_prep as prep
from torchvision import transforms, utils
import torch.nn.parallel
import numpy as np
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

__author__ = 'JosueCom'
__date__ = '5/8/2020'
__email__ = "josue.n.rivera@outlook.com"


#image_size = 500
nc = 3
ngf = 64
batch_size = 2
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

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#device = torch.device("cpu")

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

batch = next(iter(dataloader))
out = netG(batch["prev"].to(device), batch["next"].to(device))

plt.axis("off")
plt.title("Previous Training Images")

print(batch["prev"].size())
plt.imshow(np.transpose(utils.make_grid(batch["prev"].to(device)[:batch_size], padding=2, normalize=True).cpu().detach(),(1,2,0)))
plt.show()

plt.axis("off")
plt.title("Next Training Images")
print(batch["next"].size())
plt.imshow(np.transpose(utils.make_grid(batch["next"].to(device)[:batch_size], padding=2, normalize=True).cpu().detach(),(1,2,0)))
plt.show()

plt.axis("off")
plt.title("Infered Images")
print(out.size())
plt.imshow(np.transpose(utils.make_grid(out.to(device)[:batch_size], padding=2, normalize=True).cpu().detach(),(1,2,0)))
plt.show()



