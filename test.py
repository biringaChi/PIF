__author__ = 'biringaChi'
__email__ = "biringachidera@gmail.com"

import torch
import data_prep as prep
import numpy as np
import torchvision.utils as vutils

diff_pickle = open("diff.pickle","rb")
G_losses = open("G_losses.pickle","rb")
D_losses = open("D_losses.pickle","rb")

dataset = PIFDataset(root_directory=dataroot,
        diff = pickle.load(diff_pickle),
        transform=transforms.Compose([
            prep.Rescale(image_size),
            prep.ToTensor()
            ]))

dataloader = DataLoader(dataset, batch_size=batch_size,
        shuffle=True)

device = torch.device("cuda")
checkpoint = torch.load(PATH, map_location=str(device))
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.float()
model.eval()


def network_losses(G_losses, D_losses):
    plt.figure(figsize=(10,5))
    plt.title("Generator vs Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

G_losses = pickle.load(G_losses)
D_losses = pickle.load(D_losses)
network_losses(G_losses, D_losses)


def real_images(dataloader):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

real_images(dataloader)

