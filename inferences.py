from tqdm import tqdm
import os
import torch
from Unet import Unet
from torch.utils.data import DataLoader
from DataLoader import NucleiSegVal
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import gflags
from skimage import morphology, color, io, exposure
import sys

def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)


def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)


def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape[:2]
    color_mask = np.zeros((rows, cols, 3))
    boundary = morphology.dilation(gt, morphology.disk(3)) ^ gt
    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]

    img_hsv = color.rgb2hsv(img)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


if __name__ == '__main__':

    gflags.DEFINE_string('weight_path', None, 'Best Weight Path')
    gflags.DEFINE_string('id', None, 'ID')
    if gflags.FLAGS.id is None or gflags.FLAGS.weight_path is None:
        print('Provide the correct arguments')
        sys.exit()
    # Load test data
    img_size = (256, 256)

    inp_shape = (256, 256, 3)

    batch_size = 1

    # Load model
    model_path = gflags.FLAGS.id + '/Weights/' + gflags.FLAGS.weight_path + '.pth.tar'
    net = Unet(3, 1)

    net.load_state_dict(torch.load('./Weights/' + model_path))
    net.eval()

    seed = 1
    transformations_test = transforms.Compose([transforms.ToTensor()])
    test_set = NucleiSegVal(transforms=transformations_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    ious = np.zeros(len(test_loader))
    dices = np.zeros(len(test_loader))

    i = 0
    for xx, yy, name in tqdm(test_loader):
        print(xx.shape, yy.shape)
        name = name[0][:-4]
        print(name)
        pred = net(xx)
        pred = F.sigmoid(pred)
        pred = pred.detach().numpy()[0, 0, :, :]
        mask = yy.numpy()[0, 0, :, :]
        print(mask.shape, pred.shape)
        xx = xx.numpy()[0, :, :, :].transpose(1, 2, 0)
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0, 1))

        # Binarize masks
        gt = mask > 0.5
        pr = pred > 0.5

        # Remove regions smaller than 2% of the image
        # pr = remove_small_regions(pr, 0.02 * np.prod(img_size))
        if not os.path.isdir('./results'):
            os.mkdir('./results')
        io.imsave('results/{}.png'.format(name), pr * 255)
        print(gt.shape, pr.shape)
        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)

        i += 1
        if i == len(test_loader):
            break

    print('Mean IoU:', ious.mean())
    print('Mean Dice:', dices.mean())
