from .UnetModule import *

class Unet(nn.Module):
	"""docstring for Unet"""
	def __init__(self, n_ch, n_classes):
		super(Unet, self).__init__()
		self.inpt = InConv(n_ch, 64)
		self.down1 = DownConv(64, 128)
		self.down2 = DownConv(128, 256)
		self.down3 = DownConv(256, 512)
		self.up1 = UpConv(512,256)
		self.up2 = UpConv(256,128)
		self.up3 = UpConv(128,64)
		self.out = OutConv(64,n_classes)

	def forward(self, X):
		x1 = self.inpt(X)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x  = self.up1(x4, x3)
		x = self.up2(x, x2)
		x = self.up3(x, x1)
		x = self.out(x)
		return x
			