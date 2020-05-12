__author__ = 'biringaChi & JosueCom'
__email__ = ["biringachidera@gmail.com", "josue.n.rivera@outlook.com"]

import os
import pickle
from collections import OrderedDict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from data_prepocessing.ImageToArray import ImageToArrayColor

class PIFDataset(Dataset):
    def __init__(self, path, diff, transform=None, threshold = 5):
        self.path = path
        self.transform = transform
        self.groups = OrderedDict()
        self.count = 0
        self.img_num = 0
        self.buff = []
        self.diff = diff
        self.threshold = threshold # threshold to be set

        self.buff.append(self.img_num)

        for difference in self.diff:

            if difference >= self.threshold or difference <= -self.threshold:
                self.groups[self.count] = self.buff
                self.buff = []
                self.count += 1

            self.img_num += 1
            self.buff.append(self.img_num)

        self.groups[self.count] = self.buff

        items = []

        for key, values in self.groups.items():
            if len(values) < 3:
                items.append(key)

        for key in items:
            del self.groups[key]

    def __len__(self):
        sizes = []
        for key, value in self.groups.items():
            sizes.append((len(value) - 2))

        return sum(sizes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        for key, values in self.groups.items():
            temp = idx - len(values)

            if(temp > 0):
                idx = temp
            else:
                sample = {}

                sample["prev"] = ImageToArrayColor(os.path.join(self.path, "frame" + str(idx) + ".jpg"))
                sample["curr"] = ImageToArrayColor(os.path.join(self.path, "frame" + str(idx + 1) + ".jpg"))
                sample["next"] = ImageToArrayColor(os.path.join(self.path, "frame" + str(idx + 2) + ".jpg"))

                if self.transform:
                    for image_name, image_value in sample.items():
                        sample[image_name] = self.transform(image_value)

                return sample

        return None

if __name__ == '__main__':
    diff_pickle = open("metrics/planet_earth_diff.pickle","rb")

    transformed_dataset = PIFDataset(
        path='data_prepocessing/PlanetEarth',
        diff = pickle.load(diff_pickle),
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]))

    print(len(transformed_dataset))

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True)

    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch > 1:
            break
        print(sample_batched)
