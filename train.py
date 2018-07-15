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



class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count



def dice_loss(input, target):
	input = torch.sigmoid(input)
	smooth = 1

	iflat = input.view(-1)
	tflat = target.view(-1)
	intersection = (iflat * tflat).sum()

	return 1 - ((2. * intersection + smooth) /
		(iflat.sum() + tflat.sum() + smooth))


def train_net(net,train_set,test_set,epochs=5,batch_size=1,lr=0.1,val_ratio=0.1,save_model=True):
    cuda = torch.cuda.is_available()
    if cuda:
        print('Using GPU')
        net = net.cuda()
    
    # Directory for saving weights of network
    if not os.path.isdir('./Weights'):
        os.mkdir('./Weights')
        
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    
    criterion = dice_loss
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
    '''.format(epochs, batch_size, lr))
    
    print('preparing training data .....')
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    print('Done .....')
    
    print('preparing validation data .....')
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
    print('Done .....')
    
    for epoch in range(0,epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        
        # Setting network to training mode
        net.train()
        train_loss = Average()
        for i,(imgs,t_masks) in enumerate(train_loader):
            if cuda:
                imgs = imgs.cuda()
                t_masks = t_masks.cuda()
            imgs = Variable(imgs)
            t_masks = Variable(t_masks)
            p_masks = net(imgs)
            loss = criterion(p_masks,t_masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), images.size(0))
        
        # setting network to evaluation mode
        net.eval()
        val_loss = Average()
        for i,(imgs,t_masks) in enumerate(test_loader):
            if cuda:
                imgs = imgs.cuda()
                t_masks = t_masks.cuda()
            imgs = Variable(imgs)
            t_masks = Variable(t_masks)
            p_masks = net(imgs)
            loss = criterion(p_masks,t_masks)
            val_loss.update(vloss.item(), images.size(0))
        print("Epoch {}, Loss: {}, Validation Loss: {}".format(epoch + 1, train_loss.avg, val_loss.avg))
        if save_model and epoch > epochs//2:
            torch.save(net.state_dict(), './Weights/cp_{}_{}.pth.tar'.format(epoch + 1, (val_loss*batch_size)/N_test))
    
if __name__ == '__main__':

    TRAIN_PATH = './Data/Train'
    TEST_PATH = './Data/Test'

    batch_size = 4
    lr = 1e-4

    # create datasets 
    transformations_train = transforms.Compose([transforms.ToTensor()])
    transformations_val = transforms.Compose([transforms.ToTensor()])
    train_set = NucleiSegTrain(path = TRAIN_PATH, transforms=transformations_train)
    test_set = NucleiSegVal(path = TEST_PATH, transforms=transformations_val)

    net = Unet(3,1)

    train_net(net,train_set,test_set, batch_size = batch_size,lr = lr)