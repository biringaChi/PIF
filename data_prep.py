# Reference
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PIFDataset(Dataset):
    def __init__(self, root_directory, transform=None):
        self.root_directory = root_directory
        self.transform = transform

        ##### implement a grouping system from scenes (possibly hash table with scenes as key and frames as array of frame number)

    def __len__(self):
        return len(self.root_directory) #You need to add all the sizes of all the scenes and substract 2 for each scene (because we can only return in groups of tree)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join("data", "planet_earth")
        image = cv2.imread(img_name)
        sample = {'image': image} #### You need to return something like {"prev": img1, "curr": img2, "next": img3}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Rescale(object): # needs to be implements for image in sample
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image': img}   #### You have to return all the properties (one way to go about it is just updating the sample input througth teh program and returning sample)


class ToTensor(object): ## for every image in sample
    def __call__(self, sample):
        image = sample['image']
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


if __name__ == '__main__':
    transformed_dataset = PIFDataset(root_directory='data/planet_earth',
        transform=transforms.Compose([
            Rescale(256),
            ToTensor()
            ]))

    dataloader = DataLoader(transformed_dataset, batch_size=4,
        shuffle=True, num_workers=2)
        # print(next(iter(dataloader)))
