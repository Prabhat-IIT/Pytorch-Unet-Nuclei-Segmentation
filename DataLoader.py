import os
import glob
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image


class NucleiSegTrain(Dataset):
    """ Data Loader for Training data"""

    def __init__(self, path='./Data/Train/', transforms=None):
        super(NucleiSegTrain, self).__init__()
        self.path = path
        if self.path[-1] != '/':
            self.path = self.path + '/'
            self.transforms = transforms
            self.list = os.listdir(self.path + 'Images')

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        image_path = self.path + 'Images/'
        mask_path = self.path + 'Masks/'
        image = Image.open(image_path + self.list[index])
        image = image.convert('RGB')
        mask = Image.open(mask_path + self.list[index])
        mask = mask.convert('L')
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return image, mask


class NucleiSegVal(Dataset):
    """ Data Loader for Vaidation/Test data"""

    def __init__(self, path='./Data/Test/', transforms=None):
        super(NucleiSegVal, self).__init__()
        self.path = path
        if self.path[-1] != '/':
            self.path = self.path + '/'
        self.transforms = transforms
        self.list = os.listdir(self.path + 'Images')

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        image_path = self.path + 'Images/'
        mask_path = self.path + 'Masks/'
        image = Image.open(image_path + self.list[index])
        image = image.convert('RGB')
        mask = Image.open(mask_path + self.list[index])
        mask = mask.convert('L')
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        return image, mask, self.list[index]
