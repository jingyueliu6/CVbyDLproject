import torch
import torch.nn as nn

class CONV_RELU_BN_POOL(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=(1, 1), pool_size=(5, 5), pool_stride=(1, 1)):
		super(CONV_RELU_BN_POOL, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
		self.relu = nn.ReLU()
		self.bn = nn.BatchNorm2d(out_channels)
		self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)

	def forward(self, x):
		x = self.conv(x)
		x = self.relu(x)
		x = self.bn(x)
		x = self.pool(x)
		return x

class FCN(nn.Module):
	def __init__(self):
		super(FCN, self).__init__()
		self.conv1 = CONV_RELU_BN_POOL(in_channels=3, out_channels=16)
		self.conv2 = CONV_RELU_BN_POOL(in_channels=16, out_channels=32)
		self.conv3 = CONV_RELU_BN_POOL(in_channels=32, out_channels=64)
		self.conv4 = CONV_RELU_BN_POOL(in_channels=64, out_channels=64)
		self.conv5 = CONV_RELU_BN_POOL(in_channels=64, out_channels=64)
		self.conv6 = nn.Conv2d(64, 1, kernel_size=(11, 11), stride=(1, 1))
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		x = self.sigmoid(x)
		return x

class NMS(nn.Module):
	def __init__(self):
		super(NMS, self).__init__()
		self.maxpool = nn.MaxPool2d(kernel_size=(51, 51), stride=(1, 1), padding=25)

	def forward(self, x):
		y = self.maxpool(x)
		return torch.nonzero(torch.eq(x,y) & (y > 0.5), as_tuple=False)
