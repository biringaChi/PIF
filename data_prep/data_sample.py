__author__ = 'biringaChi'
__email__ = "biringachidera@gmail.com"


from __future__ import print_function
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

manualSeed = 511
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Set root directory
# dataroot = ""
batch_size = 128
image_size = 64
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                              # transforms.Resize(image_size),
                              # transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)


real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:32], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()
