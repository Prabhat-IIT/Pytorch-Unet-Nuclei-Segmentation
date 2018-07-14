import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
from skimage import io
import imgaug as ia
from imgaug import augmenters as iaa


def augment_random(patch_dir,binmask_dir, patch_size = 256):
    """Takes input patch folder name,binary mask folder name and patch_size
        creates augmented patches and mask of same patch_size, creates a folder
        and saves patches and masks in them
    """
    #target dir name
    PATH_IMG = 'aug-img' + str(patch_size) + 'x' + str(patch_size)
    PATH_BINMASK = 'aug-mask' + str(patch_size) + 'x' + str(patch_size)
    
    if patch_dir[-1] != '/':
            patch_dir = patch_dir + '/'
    if binmask_dir[-1] != '/':
            binmask_dir = binmask_dir + '/'

    # If target dir not present create it
    if not os.path.isdir(PATH_IMG):
        os.mkdir(PATH_IMG)
    if not os.path.isdir(PATH_BINMASK):
        os.mkdir(PATH_BINMASK)

    for patchname,maskname in zip(glob.glob(patch_dir + '*.png'),glob.glob(binmask_dir+'*.png')):
        with open(patchname) as p, open(maskname) as m:
            patch = io.imread(p)
            mask = io.imread(m)
            print("patch shape "+str(patch.shape))   
            print("mask shape "+str(mask.shape))
            
            ia.seed(1)    # set the seed for reproducing exact augmentation
            seq = iaa.OneOf([iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)),
            iaa.Fliplr(0.1),iaa.Flipud(0.1),
            iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25),
            iaa.AssertShape((None, patch_size, patch_size, [1, 3]))]) # apply augmenters in sequntial order
            
            aug_patch = seq.augment_images([patch])
            aug_mask = seq.augment_images([mask])
            
            io.imsave(PATH_IMG+'/'+ 'aug'+'_'+patchname[8:].replace("/","_"), aug_patch[0])
            io.imsave(PATH_BINMASK+'/'+ 'aug'+'_'+maskname[8:].replace("/","_"), aug_mask[0])
            
            
def augment_all(patch_dir,binmask_dir, patch_size = 256):
    """Takes input patch folder name,binary mask folder name and patch_size
        creates augmented patches and mask of same patch_size, creates a folder
        and saves patches and masks in them
    """
    #target dir name
    PATH_IMG = 'aug-img_all' + str(patch_size) + 'x' + str(patch_size)
    PATH_BINMASK = 'aug-mask_all' + str(patch_size) + 'x' + str(patch_size)
    
    if patch_dir[-1] != '/':
            patch_dir = patch_dir + '/'
    if binmask_dir[-1] != '/':
            binmask_dir = binmask_dir + '/'

    # If target dir not present create it
    if not os.path.isdir(PATH_IMG):
        os.mkdir(PATH_IMG)
    if not os.path.isdir(PATH_BINMASK):
        os.mkdir(PATH_BINMASK)

    for patchname,maskname in zip(glob.glob(patch_dir + '*.png'),glob.glob(binmask_dir+'*.png')):
        with open(patchname) as p, open(maskname) as m:
            patch = io.imread(p)
            mask = io.imread(m)
            print("patch shape "+str(patch.shape))   
            print("mask shape "+str(mask.shape))
            
            ia.seed(1)    # set the seed for reproducing exact augmentation
            seq1 = iaa.Sequential([iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8))]) # apply augmenters in sequntial order
            
            seq2 = iaa.Sequential([iaa.Fliplr(0.1),iaa.Flipud(0.1), 
                                   iaa.AssertShape((None, patch_size,patch_size, [1, 3]))])
            seq3 = iaa.Sequential([iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25),   
                                   iaa.AssertShape((None, patch_size, patch_size, [1, 3]))])
            seq4 = iaa.Sequential([iaa.CropAndPad(percent=(-0.25, 0.25)), 
                                  iaa.AssertShape((None, patch_size, patch_size, [1, 3]))])
            aug_patch1 = seq1.augment_images([patch])
            aug_mask1 = seq1.augment_images([mask])
            
            aug_patch2 = seq2.augment_images([patch])
            aug_mask2 = seq2.augment_images([mask])
            
            aug_patch3 = seq3.augment_images([patch])
            aug_mask3 = seq3.augment_images([mask])
            
            aug_patch4 = seq4.augment_images([patch])
            aug_mask4 = seq4.augment_images([mask])
            
            io.imsave(PATH_IMG+'/'+ 'aug_affline'+'_'+patchname[8:].replace("/","_"), aug_patch1[0])
            io.imsave(PATH_BINMASK+'/'+ 'aug_affline'+'_'+maskname[8:].replace("/","_"), aug_mask1[0])
            
            io.imsave(PATH_IMG+'/'+ 'aug_flip'+'_'+patchname[8:].replace("/","_"), aug_patch2[0])
            io.imsave(PATH_BINMASK+'/'+ 'aug_flip'+'_'+maskname[8:].replace("/","_"), aug_mask2[0])
            
            io.imsave(PATH_IMG+'/'+ 'aug_elastic'+'_'+patchname[8:].replace("/","_"), aug_patch3[0])
            io.imsave(PATH_BINMASK+'/'+ 'aug_elastic'+'_'+maskname[8:].replace("/","_"), aug_mask3[0])
            
            io.imsave(PATH_IMG+'/'+ 'aug_crop'+'_'+patchname[8:].replace("/","_"), aug_patch4[0])
            io.imsave(PATH_BINMASK+'/'+ 'aug_crop'+'_'+maskname[8:].replace("/","_"), aug_mask4[0])