import os
import glob 
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms, utils

from skimage import transform, io, img_as_float32

class NucleiSegTrain(Dataset):
	""" Data Loader for Training data"""
	def __init__(self, path='./Data/Train/', transforms=None):
		super(NucleiSeg, self).__init__()
		self.path = path
		if self.path[-1] != '/':
			self.path = self.path + '/'
		self.transforms = transforms
		self.list = os.listdir(self.path + 'Images')

	def __len__(self):
		return len(self.list)

	def __getitem__(self,idx):
		if self.path[-1] != '/':
			self.path = self.path + '/'

		IMG_PATH = self.path + 'Images/'
		MASK_PATH = self.path + 'Masks/'

		img = io.imread(IMG_PATH + self.list[idx])
		img = img_as_float32(img)
		mask = io.imread(MASK_PATH + self.list[idx])

		if self.transforms is not None:
			img = self.transforms(img)
			mask = self.transforms(mask)
		return img, mask

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

	def __getitem__(self,idx):
		if self.path[-1] != '/':
			self.path = self.path + '/'

		IMG_PATH = self.path + 'Images/'
		MASK_PATH = self.path + 'Masks/'

		img = io.imread(IMG_PATH + self.list[idx])
		img = img_as_float32(img)
		mask = io.imread(MASK_PATH + self.list[idx])

		if self.transforms is not None:
			img = self.transforms(img)
			mask = self.transforms(mask)
		return img, mask		


		