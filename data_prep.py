__author__ = 'biringaChidera'

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms

class DataPrep():
    def __init__(self, directory, batch_size):
        self.directory = directory
        self.batch_size = batch_size

    def training_data_prep(self):
        image_size = 50
        dataset = dset.ImageFolder(root = self.directory,
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle=True)
        return dataloader


if __name__ == '__main__':
    directory, batch_size = "data", 128
    training_data = DataPrep(directory, 128)
    training_batch = next(iter(training_data.training_data_prep()))
    #print(training_batch)
