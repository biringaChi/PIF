__author__ = 'biringaChidera'
__email__ = "biringaChidera@gmail.com"

import os
import glob
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
        images = []
        image_path = os.path.join(self.root_directory, '*g')
        image_files = glob.glob(image_path)
        for image in image_files:
            image = cv2.imread(image)
            images.append(image)
        curr = images[curr]
        prev = images[prev]
        sample = {"previous_image" : prev, "current_image" : curr}
        curr += 1
        prev += 1
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        for image_name, image_value in sample.items():
            image = image_value.transpose((2, 0, 1))
        return torch.from_numpy(image)


if __name__ == '__main__':
    transformed_dataset = PIFDataset(root_directory='scratch/1',
        transform=transforms.Compose([
            #Rescale(256),
            ToTensor()
            ]))

    dataloader = DataLoader(transformed_dataset, batch_size=100,
        shuffle=True, num_workers=2)
    print(next(iter(dataloader)))
