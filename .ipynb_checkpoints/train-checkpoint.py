from __future__ import print_function
import random
import torch
import pickle
import torch.nn as nn
import data_prep as prep
from torchvision import transforms, utils
import torch.nn.parallel
import torch.optim as optim
import numpy as np
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader


__author__ = 'JosueCom'
__date__ = '4/30/2020'
__email__ = "josue.n.rivera@outlook.com"


dataroot = 'data_prepocessing/PlanetEarth'

batch_size = 10
image_size = 500
nc = 3
nz = 100
ngf = 25
ndf = 25
num_epochs = 3
lr = 0.0002
beta1 = 0.5
ngpu = torch.cuda.device_count()
lf_to_rg_ratio = 0.5

#stats
G_losses = []
D_losses = []

diff_pickle = open("planet_earth_diff.pickle","rb")

print("> loading dataset")
dataset = prep.PIFDataset(
    path='data_prepocessing/PlanetEarth',
    diff = pickle.load(diff_pickle),
    transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("Done loading dataset")

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

## generator
print("> loading generator")
"""netG = Generator(ngpu, nc, ngf).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)"""

netG = torch.load("nice/generator6.bin")

print("Done loading generator")

"""print("> loading discriminator")
## discriminator
netD = Discriminator(ngpu, nc, ndf).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)
print("Done loading discriminator")"""

# which of the two input is the real one for the discriminator
true_left = 0
true_right = 1

#optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


#loss = nn.BCELoss()
loss = nn.MSELoss()
print("> training")
torch.cuda.empty_cache()
len_dt = len(dataloader)
for epoch in range(num_epochs):

    for i, data in enumerate(dataloader, 0):
        """
        netD.zero_grad()
        b_size = data["prev"].size(0)
        cur = data["curr"].to(device)

       	######### train discriminator
        gen_out = netG(data["prev"].to(device), data["next"].to(device))
        torch.cuda.empty_cache()
        
        ## Train with correct output on the left side
        label = torch.full((b_size,), true_left, device=device, dtype=torch.float32)
        dis_out = netD(cur, gen_out.detach()).view(-1)
        torch.cuda.empty_cache()

        errD_left = loss(dis_out, label)
        errD_left.backward(retain_graph=True)
        D_left = dis_out.mean().item()

        ## Train with correct output on the right side
        label.fill_(true_right)
        dis_out = netD(gen_out.detach(), cur).view(-1)
        torch.cuda.empty_cache()

        errD_right = loss(dis_out, label)
        errD_right.backward()
        D_right = dis_out.mean().item()

        errD = errD_left + errD_right

        optimizerD.step()


       	######### train generator (check again)
        netG.zero_grad()
        side = random.randint(0, 1)
        label.fill_(side)
    
        if(side):
            dis_out = netD(cur, gen_out).view(-1)
        else:
            dis_out = netD(gen_out, cur).view(-1)

        errG = loss(dis_out, label)  ## where loss is applied

        errG.backward()
        D_gen = dis_out.mean().item()

        optimizerG.step()

        #### stats
        if i % 20 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD_left: %.4f\tD_right: %.4f\tD_gen: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_left, D_right, D_gen))

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        torch.cuda.empty_cache()
        """
        netG.zero_grad()
        gen_out = netG(data["prev"].to(device), data["next"].to(device))
        torch.cuda.empty_cache()
        
        errG = loss(gen_out, data["curr"].to(device))
        errG.backward()
        D_gen = gen_out.mean().item()
        
        optimizerG.step()
        
        G_losses.append(errG.item())
        
        if i % 20 == 0:
            print('[%d/%d][%d/%d]\tLoss_G: %.4f\tD_gen: %.4f'
                  % (epoch, num_epochs, i, len_dt, errG.item(), D_gen))
        
    
    torch.save(netG, "nice/generator6.bin")
    #torch.save(netD, "nice/discriminator4.bin")
    
print("Done training")

#torch.save(netG, "nice/generator5.bin")
#torch.save(netD, "nice/discriminator5.bin")

# store network stats for testing
G = open("nice/G_losses6_2.pickle","wb")
pickle.dump(G_losses, G)
G.close()

#D = open("nice/D_losses4.pickle","wb")
#pickle.dump(D_losses, D)
#D.close()

