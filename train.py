import numpy as np
import torch # work as numpy here
import torch.autograd as autograd # builds computational graph  
import torch.nn as nn # neural net library
import torch.nn.functional as F # all the non linearities
import torch.optim as optim # optimization package
from skimage import transform, io
from sklearn.model_selection import train_test_split
from itertools import chain

"""def dice_loss(input,target):
    '''input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input'''
    
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques=np.unique(target.numpy())
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs=F.sigmoid(input)
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)
    

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)
    

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=2*(num/(den1+den2))
    dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

    return dice_total """

def dice_loss(input, target):
	input = torch.sigmoid(input)
	smooth = 1

	iflat = input.view(-1)
	tflat = target.view(-1)
	intersection = (iflat * tflat).sum()

	return 1 - ((2. * intersection + smooth) /
		(iflat.sum() + tflat.sum() + smooth))

def batch(iterable1,batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable1):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield np.array(b)
            b = []

    if len(b) > 0:
        yield np.array(b)

def train_net(net,X,Y,epochs=5,batch_size=1,lr=0.1,val_ratio=0.1):
    X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=val_ratio)
    
    N_train = len(X_train)
    
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    
    criterion = dice_loss
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
    '''.format(epochs, batch_size, lr, N_train,
               len(X_val)))
    for epoch in range(0,epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        
        epoch_loss = 0
        
        for i,(imgs,t_masks) in enumerate(zip(batch(X_train, batch_size), batch(Y_train, batch_size))):
            #converting to N*Ch*H*W from N*H*W*ch
            imgs = np.moveaxis(imgs,3,1)
            t_masks = np.moveaxis(t_masks,3,1)

            #converting to pytorch tensor
            imgs = torch.from_numpy(imgs)
            t_masks = torch.from_numpy(t_masks)
            if i == 0:
            	print(imgs.shape)

            p_masks = net(imgs)
            
            loss = criterion(p_masks,t_masks)
            epoch_loss += loss
            
            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train , loss ))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print('Epoch finished ! Loss: {}'.format(epoch_loss))
        
        val_dice = criterion(net(X_val),Y_val)
        print('Validation Dice Coeff: {}'.format(val_dice))                