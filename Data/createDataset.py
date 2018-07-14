from patches import createImgPatches, createBinMask
import shutil
import os
from sklearn.model_selection import train_test_split

IMG_DIR = './train-img256x256/'
BIN_MASK_DIR = './train-bin-mask256x256/'

source = '../segmentation_training_set/'
patch_shape = 256

if not os.path.isdir(IMG_DIR):
	createBinMask(source)
	createImgPatches(source, patch_shape)

# img and mask files both have same names just different directories
img_files = os.listdir('./train-img256x256')	

train_files , test_files = train_test_split(img_files,test_size = 0.1)

for file in train_files:
	shutil.move(IMG_DIR + file, 'Train/Images/')
	shutil.move(BIN_MASK_DIR + file, 'Train/Masks/')

for file in test_files:
	shutil.move(IMG_DIR + file, 'Test/Images/')
	shutil.move(BIN_MASK_DIR + file , 'Test/Masks/')

print('Done')
		

