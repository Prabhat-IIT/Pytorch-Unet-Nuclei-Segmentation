import numpy as np

def pad_zeros(img, height, width, channels):
    """Pads img (with 0's) to make it fit for extracting patches of shape height*width from it"""
    print('input shape {}'.format(img.shape))
    h = 0 if img.shape[0]%height == 0 else height - img.shape[0]%height
    w = 0 if img.shape[1]%width == 0 else width - img.shape[1]%width
    pad_shape = tuple(np.zeros((len(img.shape),2),dtype=img.dtype))
    pad_shape = [tuple(x) for x in pad_shape]
    pad_shape[0] = (0,h)
    pad_shape[1] = (0,w)
    img = np.pad(img,pad_shape,mode='constant')
    print('output shape {}'.format(img.shape))
    return img