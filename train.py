import torch # work as numpy here
import torch.autograd as autograd # builds computational graph
from torch.autograd import Variable # wraps tensors and helps compute gradient
import torch.nn as nn # neural net library
import torch.nn.functional as F # all the non linearities
import torch.optim as optim # optimization package
from torch.utils.data import DataLoader

import os
import numpy as np
from Unet import Unet
from DataLoader import NucleiSegTrain, NucleiSegVal
from skimage import transform, io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import GPUtil




class History(object):
    """contains History of model like train losses and val losses"""
    def __init__(self,train_loss=None,val_loss=None):
        super(History, self).__init__()
        self.train_loss = train_loss
        self.val_loss = val_loss
    
    def get_train_loss(self):
        return self.train_loss
    
    def get_val_loss(self):
        return self.val_loss
    
    def plot_loss(self):
        plt1, = plt.plot(self.train_loss, 'b-o', label="train loss")
        plt2, = plt.plot(self.val_loss, 'g-o', label="test loss")
        plt.legend([plt1, plt2], ['Train Loss', 'Val Loss'])
        plt.show()

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

def train_net(net,train_set,test_set,epochs=5,batch_size=1,lr=0.1,val_ratio=0.1,save_model=True):
    cuda = torch.cuda.is_available()
    if cuda:
        print('Using GPU')
        net = net.cuda()
    
    # Directory for saving weights of network
    if not os.path.isdir('./Weights'):
        os.mkdir('./Weights')
        
    N_train = len(train_set)
    N_test = len(test_set)
    
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
               N_test))
    
    print('preparing training data .....')
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    print('Done .....')
    
    print('preparing validation data .....')
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
    print('Done .....')
    
    train_loss = []
    test_loss = []
    for epoch in range(0,epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        
        # Setting network to training mode
        net.train()
        
        epoch_loss = 0
        
        for i,(imgs,t_masks) in enumerate(train_loader):
            if cuda:
                imgs = imgs.cuda()
                t_masks = t_masks.cuda()
            if i==0:
                print(imgs.shape)
                print(t_masks.shape)
            imgs = Variable(imgs)
            t_masks = Variable(t_masks)

            p_masks = net(imgs)
            
            loss = criterion(p_masks,t_masks)
            epoch_loss += loss
            
            #print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train , loss ))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss.append(epoch_loss.cpu().detach().numpy())
        print('Epoch finished ! Loss: {}'.format((epoch_loss*batch_size)/N_train))
        
        print(GPUtil.showUtilization())
        # setting network to evaluation mode
        net.eval()
        
        val_loss = 0
        for i,(imgs,t_masks) in enumerate(test_loader):
            if i==0:
                print(imgs.dtype,imgs.shape)
                print(t_masks.dtype,t_masks.shape)
            if cuda:
                imgs = imgs.cuda()
                t_masks = t_masks.cuda()
                
            imgs = Variable(imgs)
            t_masks = Variable(t_masks)
            
            p_masks = net(imgs)
            
            loss = criterion(p_masks,t_masks).cpu()
            val_loss += loss.detach().numpy()
            
        test_loss.append(val_loss)    
        print('Validation Dice Loss: {}'.format((val_loss*batch_size)/N_test))
        
        if save_model and epoch > epochs//2:
            torch.save(net.state_dict(), './Weights/cp_{}_{}.pth.tar'.format(epoch + 1, (val_loss*batch_size)/N_test))
    Hist = History(train_loss, test_loss)
    Hist.plot_loss()
    
if __name__ == '__main__':

    TRAIN_PATH = './Data/Train'
    TEST_PATH = './Data/Test'

    batch_size = 4
    lr = 1e-4

    # create datasets 
    train_set = NucleiSegTrain(path = TRAIN_PATH, transforms=None)
    test_set = NucleiSegVal(path = TEST_PATH, transforms=None)

    net = Unet(3,1)

    train_net(net,train_set,test_set, batch_size = batch_size,lr = lr)