__author__ = 'biringaChidera'
__email__ = "biringaChidera@gmail.com"

import os
import cv2
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class PIFDataset(Dataset):
    def __init__(self, root_directory, transform=None):
        self.root_directory = root_directory
        self.transform = transform

    def __len__(self):
        size = []
        for filename in os.listdir(self.root_directory):
            if filename.endswith(".jpg"):
                path = os.path.join(self.root_directory, filename)
                img = cv2.imread(path)
                size.append(img)
            else:
                continue
        return len(size) - 2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        prev = 0
        curr = 1
        nex = 2
        images = []
        image_path = os.path.join(self.root_directory, '*g')
        image_files = glob.glob(image_path)
        for image in image_files:
            image = cv2.imread(image)
            images.append(image)
        curr = images[curr]
        prev = images[prev]
        nex = images[nex]
        sample = {"previous_image": prev, "current_image": curr, "next_image": nex}
        curr += 1
        prev += 1
        nex += 1
        if self.transform:
            sample = self.transform(sample)
        return sample


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        for image_name, image_value in sample.items():
            h, w = image_value.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size
            new_h, new_w = int(new_h), int(new_w)
            image = cv2.resize(image_value, (new_h, new_w))
            sample[image_name] = image
        return sample


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        for image_name, image_value in sample.items():
            h, w = image_value.shape[:2]
            new_h, new_w = self.output_size
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            image = image_value[top: top + new_h, left: left + new_w]
            sample[image_name] = image
        return sample


class ToTensor(object):
    def __call__(self, sample):
        for image_name, image_value in sample.items():
            image = image_value.transpose((2, 0, 1))
            sample[image_name] = torch.from_numpy(image)
        return sample


if __name__ == '__main__':
    transformed_dataset = PIFDataset(
        root_directory='scratch/1',
        transform=transforms.Compose([
            Rescale(256),
            RandomCrop(224),
            ToTensor()]))

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True)
#    print(next(iter(dataloader)))
