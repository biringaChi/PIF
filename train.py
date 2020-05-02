__author__ = 'JosueCom'
__date__ = '4/30/2020'
__email__ = "josue.n.rivera@outlook.com"

from __future__ import print_function
import random
import torch
import pickle
import torch.nn as nn
import data_prep as prep
from torchvision import transforms
import torch.nn.parallel
import torch.optim as optim
import numpy as np
from generator import Generator
from discriminator import Discriminator

dataroot = "data_prepocessing/Popeye2"

batch_size = 128
image_size = 500
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1
lf_to_rg_ratio = 0.5

#stats
G_losses = []
D_losses = []

diff_pickle = open("diff.pickle","rb")
dataset = PIFDataset(root_directory=dataroot,
        diff = pickle.load(diff_pickle),
        transform=transforms.Compose([
            prep.Rescale(image_size),
            prep.ToTensor()
            ]))

dataloader = DataLoader(dataset, batch_size=batch_size,
        shuffle=True)

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

## discriminator
netD = Discriminator(ngpu, nc, ndf).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)

# which of the two input is the real one for the discriminator
true_left = 0
true_right = 1

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


loss = nn.BCELoss()

for epoch in range(num_epochs):
   
    for i, data in enumerate(dataloader, 0):

        netD.zero_grad()
        b_size = data["prev"].size(0)

        cut = (int) lf_to_rg_ratio * b_size
       	
       	######### train discriminator
        gen_out = netG(data["prev"], data["next"])

        ## Train with correct output on the left side
        label = torch.full((cut,), true_left, device=device)
        dis_out = netD(data["curr"][:cut].detach(), gen_out[:cut]).view(-1)

        errD_left = loss(dis_out, label)
        errD_left.backward()
        D_left = dis_out.mean().item()
        
        ## Train with correct output on the right side
        label = torch.full((b_size - cut,), true_right, device=device)
        dis_out = netD(gen_out[cut:], data["curr"][cut:].detach()).view(-1)

        errD_right = loss(dis_out, label)
        errD_right.backward()
        D_right = dis_out.mean().item()

        errD = errD_left + errD_right

        optimizerD.step()

       	
       	######### train generator (check again)
        netG.zero_grad()
        label = torch.full((cut,), true_right, device=device)
        label = torch.cat((label, torch.full((cut,), true_left, device=device)), dim=1);
       
        dis_out = netD(gen_out).view(-1)
       
        errG = loss(dis_out, label)  ## where loss is applied
       
        errG.backward()
        D_gen = dis_out.mean().item()
       
        optimizerG.step()

        #### stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD_left: %.4f\tD_right: %.4f\tD_right: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_left, D_right, D_gen))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

torch.save(netG.state_dict(), "generator.bin")
torch.save(netD.state_dict(), "discriminator.bin")