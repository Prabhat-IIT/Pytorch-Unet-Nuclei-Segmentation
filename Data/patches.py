import os
import glob
import numpy as np
from skimage import io
from patchHelper import extract_patches, pad_zeros
from preprocess import batchStainNormalize



def createImgPatches(source,patch_size,normalize = True):
    """Takes input source folder name and patch_size
        creates patches of given patch_size, creates a folder
        and saves patches in them
    """
    PATH_IMG = "train-img"
    PATH_MASK = "train-mask"
    PATH_BINMASK = "train-bin-mask"
    #target dir name
    PATH_IMG = PATH_IMG + str(patch_size) + 'x' + str(patch_size)
    PATH_MASK = PATH_MASK + str(patch_size) + 'x' + str(patch_size)
    PATH_BINMASK = PATH_BINMASK + str(patch_size) + 'x' + str(patch_size)

    if source[-1] != '/':
            source = source + '/'

    # If target dir not present create it
    if not os.path.isdir(PATH_IMG):
        os.mkdir(PATH_IMG)
    if not os.path.isdir(PATH_MASK):
        os.mkdir(PATH_MASK)
    if not os.path.isdir(PATH_BINMASK):
        os.mkdir(PATH_BINMASK)


    for i in range(1,16):
        if i < 10:
            i = '0' + str(i)
        print(i)
    
        img_path = source + 'image' + str(i) + '.png'
        #mask_path = source + 'image' + str(i) + '_poly.png'   
        binmask_path = source + 'image' + str(i) + '_maskbin.png'

        if normalize:
            img = pad_zeros(batchStainNormalize(io.imread(img_path)[:,:,:3]),patch_size,patch_size,3)
        else:    
            img = pad_zeros(io.imread(img_path)[:,:,:3],patch_size,patch_size,3)
        #mask = pad_zeros(io.imread(mask_path)[:,:,:3],patch_size,patch_size,3)
        binmask = pad_zeros(io.imread(binmask_path),patch_size,patch_size,1)

        img_patches = extract_patches(img,(patch_size,patch_size))
        #mask_patches = extract_patches(mask,(patch_size,patch_size))
        binmask_patches = extract_patches(binmask,(patch_size,patch_size))

        for j,(img, binmask) in enumerate(zip(img_patches, binmask_patches)):
            io.imsave(PATH_IMG+'/'+ str(i)+'_'+str(j)+'.png', img)
            #io.imsave(PATH_MASK+'/'+ str(i)+'_'+str(j)+'.png', mask)
            io.imsave(PATH_BINMASK+'/'+ str(i)+'_'+str(j)+'.png', binmask)

def createBinMask(source):
    """ Takes .txt file from source folder and
        creates a binary mask and save it there
    """
    if source[-1] != '/':
            source = source + '/'

    for filename in glob.glob(source + '*.txt'):
        with open(filename) as f:
            content = f.readlines()
            shape = content[0].strip().split()
            shape = [int(x) for x in shape]
            shape = [shape[1],shape[0]]
            print(shape)
            content = content[1:]
            content = [bool(int(x.strip().split()[0]))*255 for x in content]
            content = np.array(content).reshape(shape)
            io.imsave(filename[:-4] + 'bin.png',content)

if __name__ == "__main__":
    source = '../segmentation_training_set/'
    shape = 256
    
    createBinMask(source)
    createImgPatches(source, shape)