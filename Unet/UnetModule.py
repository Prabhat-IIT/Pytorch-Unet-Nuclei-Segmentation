import torch # work as numpy here
import torch.autograd as autograd # builds computational graph  
import torch.nn as nn # neural net library
import torch.nn.functional as F # all the non linearities
import torch.optim as optim # optimization package

class Dconv(nn.Module):
	"""Basic block of Conv2d, BatchNorm2d, and Relu layers conneted togather twice"""
	def __init__(self, In_ch, Out_ch, K_size=3, stride=1, padding=1):
		super(Dconv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(In_ch, Out_ch, K_size, padding=1),
			nn.BatchNorm2d(Out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(Out_ch, Out_ch, K_size, padding=1),
			nn.BatchNorm2d(Out_ch),
			nn.ReLU(inplace=True)
			)

	def forward(self,X):
		return self.conv(X)

class InConv(nn.Module):
	"""Convolution layer for the input to Unet"""
	def __init__(self, In_ch, Out_ch):
		super(InConv, self).__init__()
		self.conv = Dconv(In_ch, Out_ch)

	def forward(self, X):
		return self.conv(X)

class DownConv(nn.Module):
	"""Block of layers stacked up togather for Down Convolution"""
	def __init__(self, In_ch, Out_ch):
		super(DownConv, self).__init__()
		self.conv = nn.Sequential(
			nn.MaxPool2d(2),
			Dconv(In_ch, Out_ch)
			)

	def forward(self, X):
		return self.conv(X)			
					

class UpConv(nn.Module):
	"""Block of layers stacked up togather for Up Convolution"""
	def __init__(self, In_ch, Out_ch, learnable=True):
		super(UpConv, self).__init__()
		
		# learnable -> parameter to specify if to learn Upsampling or Use extrapolation
		if learnable == False:
			self.up = nn.Sequential(
				nn.Conv2d(In_ch, In_ch//2, kernel_size=2,padding=2),
				nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
				)
		else:
			# In_ch = In_ch//2 since before convloution input has In_ch//2 number of channels
			# Out_ch = In_ch//2 since upsampling or transpose convolution doesnot alter count of channels
			self.up = nn.ConvTranspose2d(In_ch, In_ch//2, kernel_size=2, stride=2)	

		self.conv = Dconv(In_ch,Out_ch)

	def forward(self, X1, X2):
		# X1 input from below X2 input from left
		X1 = self.up(X1)

		# spatial size of X1 < spatial size of X2
		diffX, diffY = (X2.size()[2] - X1.size()[2], X2.size()[3] - X1.size()[3])
		X1 = F.pad(X1, (diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2))
		X = torch.cat([X2,X1], dim=1)
		return self.conv(X)

class OutConv(nn.Module):
	"""Final Output layer with kernel size = 1"""
	def __init__(self, In_ch, Out_ch):y
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(In_ch, Out_ch,1)

	def forward(self, X):
		return self.conv(X)