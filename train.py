import torch  # work as numpy here
from torch.autograd import Variable  # wraps tensors and helps compute gradient
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import os
from tensorboardX import SummaryWriter
from Unet import Unet
from DataLoader import NucleiSegTrain, NucleiSegVal
import gflags
import sys


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


class SoftDiceLoss(nn.Module):
    '''
    Soft Dice Loss
    '''

    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def train_net(net, train_set, test_set, epochs=40, batch_size=1, lr=0.1, val_ratio=0.1, save_model=True, id=None):
    writer = SummaryWriter()
    cuda = torch.cuda.is_available()
    print(id, 'ID')
    if cuda:
        print('Using GPU')
        net = net.cuda()
    if not os.path.isdir(id):
        os.mkdir(id)
    weight_path = id + '/Weights/'
    # Directory for saving weights of network
    if not os.path.isdir(weight_path):
        os.mkdir(weight_path)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    criterion1 = SoftDiceLoss().cuda()
    criterion2 = nn.BCEWithLogitsLoss().cuda()
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
    '''.format(epochs, batch_size, lr))

    print('preparing training data .....')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print('Done .....')
    print('preparing validation data .....')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print('Done .....')
    train_global_step = 0
    val_global_step = 0
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        # Setting network to training mode
        net.train()
        train_loss = Average()
        b_c_e_loss = Average()
        dice_loss = Average()
        for i, (imgs, t_masks) in enumerate(train_loader):
            if cuda:
                imgs = imgs.cuda()
                t_masks = t_masks.cuda()
            imgs = Variable(imgs)
            t_masks = Variable(t_masks)
            p_masks = net(imgs)
            loss1 = criterion1(p_masks, t_masks)
            loss2 = criterion2(p_masks, t_masks)
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), imgs.size(0))
            b_c_e_loss.update(loss2.item(), imgs.size(0))
            dice_loss.update(loss1.item(), imgs.size(0))
            train_global_step += 1
            writer.add_scalar('Train Total Loss', loss, train_global_step)
            writer.add_scalar('Train BCE Loss', loss2, train_global_step)
            writer.add_scalar('Train Dice Loss', loss1, train_global_step)

        # setting network to evaluation mode
        net.eval()
        val_loss = Average()
        b_c_e_loss_val = Average()
        dice_loss_val = Average()
        for i, (imgs, t_masks, name) in enumerate(test_loader):
            if cuda:
                imgs = imgs.cuda()
                t_masks = t_masks.cuda()
            imgs = Variable(imgs)
            t_masks = Variable(t_masks)
            p_masks = net(imgs)
            vloss1 = criterion1(p_masks, t_masks)
            vloss2 = criterion2(p_masks, t_masks)
            vloss = vloss1 + vloss2
            val_loss.update(vloss.item(), imgs.size(0))
            b_c_e_loss_val.update(vloss2.item(), imgs.size(0))
            dice_loss_val.update(vloss1.item(), imgs.size(0))
            val_global_step += 1
            writer.add_scalar('Val Total Loss', vloss, val_global_step)
            writer.add_scalar('Val BCE Loss', vloss2, val_global_step)
            writer.add_scalar('Val Dice Loss', vloss1, val_global_step)
            if epoch % 10 == 0:
                writer.add_image('Validation GT', t_masks, epoch)
                writer.add_image('Validation Prediction', p_masks, epoch)
                writer.add_image('Validation Input Photos', imgs, epoch)

        print(
            "Epoch {}, Total Train Loss: {},BCE Train Loss: {}, Dice Train Loss: {}, Validation Total loss: {}, Validation BCE: {}, Validation Dice: {}".format(
                epoch + 1, train_loss.avg, b_c_e_loss.avg, dice_loss.avg, val_loss.avg, b_c_e_loss_val.avg,
                dice_loss_val.avg))
        if save_model and epoch > epochs // 2:
            torch.save(net.state_dict(), weight_path + '{}.pth.tar'.format(epoch + 1))


if __name__ == '__main__':
    gflags.DEFINE_string('id', None, 'ID for experiment')
    gflags.DEFINE_string('path', 'Data/', 'Path for Dataset')
    gflags.FLAGS(sys.argv)
    TRAIN_PATH = gflags.FLAGS.path + 'Train'
    TEST_PATH = gflags.FLAGS.path + 'Test'

    batch_size = 4
    lr = 1e-4

    # create datasets 
    transformations_train = transforms.Compose([transforms.ToTensor()])
    transformations_val = transforms.Compose([transforms.ToTensor()])
    train_set = NucleiSegTrain(path=TRAIN_PATH, transforms=transformations_train)
    test_set = NucleiSegVal(path=TEST_PATH, transforms=transformations_val)

    net = Unet(3, 1)

    train_net(net, train_set, test_set, batch_size=batch_size, lr=lr, id=gflags.FLAGS.id)

