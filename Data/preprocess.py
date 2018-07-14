import os 
import glob
from skimage import io, transform, img_as_float32, img_as_ubyte
from staintools import ReinhardNormalizer,VahadaneNormalizer,MacenkoNormalizer
from staintools.utils.visual import patch_grid, build_stack

print('hello world')

TARGET_IMG_PATH = '../he.png'

target = io.imread(TARGET_IMG_PATH)

R = ReinhardNormalizer()
V = VahadaneNormalizer()
M = MacenkoNormalizer()

R.fit(target)
V.fit(target)
M.fit(target)


def stainNormalize(img,method = 'V'):
	"""Expexts an img with dtype uint8 and method in ['V','M','R']
	   Returns stain normalized image in uint8 (ubyte) format
	   Expected input format --> H*W*C
	"""
	normalizer = V
	if method == 'V':
		normalizer = V

	if method == 'R':
		normalizer = R

	if method == 'M':
		normalizer = M

	return normalizer.transform(img)	

def batchStainNormalize(batch, method = 'V'):
	"""Expects a Batch or List of images in either uint8 (ubyte) or float32 dtype
	   Preserves the dtype and returns a batch with stain normalized images
	   Expected input format of images --> H*W*C
	"""
	float32 = False;
	if batch[0].dtype == 'float32':
		float32 = True
	for i,img in enumerate(batch):	
		img = stainNormalize(img_as_ubyte(img),method)
		if float32:
			img = img_as_float32(img)
		batch[i] = img	
	return batch	

if __name__ == '__main__':
	print('hello world')

	