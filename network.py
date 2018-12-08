import os
import json
import argparse
import numpy as np
import scipy
from random import shuffle
import time
import sys
import pdb
from collections import defaultdict
import itertools
# pytorch imports
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data.sampler as samplers
import imageio
from skimage.io import imread, imsave
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets
from skimage import io, transform
from PIL import Image
from utils import ssim
import torchvision.transforms.functional as TF
import torchvision
import numpy.random as random
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.downconv = torch.nn.Sequential(
			nn.Conv2d(1,3,4,stride=2,padding=1),
			torch.nn.BatchNorm2d(3),
			torch.nn.ReLU(),
			)
		self.conv = torch.nn.Sequential(
			#2x64x64
			nn.Conv2d(4,16,4,stride=2,padding=1),
			torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(),
			#16x32x32
			nn.Conv2d(16,32,4,stride=4,padding=1),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			#32x8x8
			nn.Conv2d(32,128,4,stride=4,padding=1),
			#128x2x2
		)
		self.fc =  torch.nn.Linear(128*2*2, 1, bias=False)
	def forward(self, x128, x64):
		x = self.downconv(x128)
		x = torch.cat((x,x64),1)
		x = self.conv(x)
		x = x.view(x.size(0),-1)
		x = self.fc(x)
		x = x.view(-1,1)
		return x

class Network_res_128(nn.Module):
	def __init__(self):
		super(Network_res_128, self).__init__()
		
		p=0.01
		self.conv1 = torch.nn.Sequential(
			#64x64x1
			nn.Conv2d(1,16,3,stride=2,padding=1,dilation=1),
			torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#32x32x16
			nn.Conv2d(16,32,3,stride=2,padding=2,dilation=2),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#32x32x16
			nn.Conv2d(32,64,3,stride=2,padding=3,dilation=3),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#64x8x8
			#8x8x64
			nn.ConvTranspose2d(64,32,4,stride=2,padding=1),
			# torch.nn.LayerNorm((32, 16, 16)),
			torch.nn.BatchNorm2d(32),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#16x16x32
			nn.ConvTranspose2d(32,32,4,stride=2,padding=1),
			# torch.nn.LayerNorm((32, 32, 32)),
			torch.nn.BatchNorm2d(32),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#16x32x32
			nn.ConvTranspose2d(32,16,4,stride=2,padding=1),
			# torch.nn.LayerNorm((16, 64, 64)),
			torch.nn.BatchNorm2d(16),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#4x64x64
			nn.ConvTranspose2d(16,8,3,stride=1,padding=1),
			# torch.nn.LayerNorm((8, 128, 128)),
			torch.nn.BatchNorm2d(8),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#2x128x128
			nn.ConvTranspose2d(8,4,3,stride=1,padding=1),
			torch.nn.LeakyReLU(0.2),
			#2x128x128
			nn.ConvTranspose2d(4,1,3,stride=1,padding=1),
			# torch.nn.Sigmoid(),
			)

		self.conv2 = torch.nn.Sequential(
			#1x64x64
			nn.ConvTranspose2d(1,2,3,stride=1,padding=1),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			# torch.nn.Sigmoid(),
			)


		self.conv_mixer = torch.nn.Sequential(
			#2x128x128
			nn.Conv2d(3,2,3,stride=1,padding=1),
			torch.nn.ReLU(),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			# torch.nn.Sigmoid(),
			#1x128x128
			)

	def forward(self, x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(x)
		# x = (conv1+1)/2
		x = torch.cat((x,conv1,conv2),1)
		x = self.conv_mixer(x)
		return x

class Network_res(nn.Module):
	def __init__(self):
		super(Network_res, self).__init__()
		
		p = 0.01
		self.conv1 = torch.nn.Sequential(
			#64x64x1
			nn.Conv2d(1,16,3,stride=2,padding=1,dilation=1),
			torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#32x32x16
			nn.Conv2d(16,32,3,stride=2,padding=2,dilation=2),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#32x32x16
			nn.Conv2d(32,64,3,stride=2,padding=3,dilation=3),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#64x8x8
			#8x8x64
			nn.ConvTranspose2d(64,32,4,stride=2,padding=1),
			# torch.nn.LayerNorm((32, 16, 16)),
			torch.nn.BatchNorm2d(32),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#16x16x32
			nn.ConvTranspose2d(32,32,4,stride=2,padding=1),
			# torch.nn.LayerNorm((32, 32, 32)),
			torch.nn.BatchNorm2d(32),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#16x32x32
			nn.ConvTranspose2d(32,16,4,stride=2,padding=1),
			# torch.nn.LayerNorm((16, 64, 64)),
			torch.nn.BatchNorm2d(16),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#4x64x64
			nn.ConvTranspose2d(16,8,3,stride=1,padding=1),
			# torch.nn.LayerNorm((8, 128, 128)),
			torch.nn.BatchNorm2d(8),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#2x128x128
			nn.ConvTranspose2d(8,4,3,stride=1,padding=1),
			torch.nn.LeakyReLU(0.2),
			#2x128x128
			nn.ConvTranspose2d(4,1,3,stride=1,padding=1),
			# torch.nn.Sigmoid(),
			)

		self.conv2 = torch.nn.Sequential(
			#1x64x64
			nn.ConvTranspose2d(1,2,3,stride=1,padding=1),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			# torch.nn.Sigmoid(),
			)


		self.conv_mixer = torch.nn.Sequential(
			#2x128x128
			nn.Conv2d(3,2,3,stride=1,padding=1),
			torch.nn.ReLU(),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			# torch.nn.Sigmoid(),
			#1x128x128
			)

	def forward(self, x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(x)
		# x = (conv1+1)/2
		x = torch.cat((x,conv1,conv2),1)
		x = self.conv_mixer(x)
		return x

class Network_up(nn.Module):
	def __init__(self):
		super(Network_up, self).__init__()
		
		p = 0.01
		self.conv1 = torch.nn.Sequential(
			#64x64x1
			nn.Conv2d(2,16,3,stride=2,padding=1,dilation=1),
			torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#32x32x16
			nn.Conv2d(16,32,3,stride=2,padding=2,dilation=2),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#32x32x16
			nn.Conv2d(32,64,3,stride=2,padding=3,dilation=3),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#64x8x8
			#8x8x64
			nn.ConvTranspose2d(64,32,4,stride=2,padding=1),
			torch.nn.LayerNorm((32, 16, 16)),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#16x16x32
			nn.ConvTranspose2d(32,32,4,stride=2,padding=1),
			torch.nn.LayerNorm((32, 32, 32)),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#16x32x32
			nn.ConvTranspose2d(32,16,4,stride=2,padding=1),
			torch.nn.LayerNorm((16, 64, 64)),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#4x64x64
			nn.ConvTranspose2d(16,8,4,stride=2,padding=1),
			torch.nn.LayerNorm((8, 128, 128)),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#2x128x128
			nn.ConvTranspose2d(8,4,3,stride=1,padding=1),
			torch.nn.LeakyReLU(0.2),
			#2x128x128
			nn.ConvTranspose2d(4,1,3,stride=1,padding=1),
			torch.nn.Sigmoid(),
			)

		self.conv2 = torch.nn.Sequential(
			#1x64x64
			nn.ConvTranspose2d(2,2,4,stride=2,padding=1),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			torch.nn.Sigmoid(),
			)


		self.conv_mixer = torch.nn.Sequential(
			#2x128x128
			nn.Conv2d(6,2,3,stride=1,padding=1),
			torch.nn.ReLU(),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			torch.nn.Sigmoid(),
			#1x128x128
			)

	def forward(self, x):
		conv2 = self.conv2(x)
		conv1 = self.conv1(x)
		bilinear = nn.functional.interpolate(x,scale_factor=2, mode='bilinear', align_corners = False)
		nearest = nn.functional.interpolate(x,scale_factor=2, mode='nearest')
		# x = self.conv0(x)
		# x = (conv1+1)/2
		x = torch.cat((bilinear,nearest,conv1,conv2),1)
		x = self.conv_mixer(x)
		return x


class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		
		p = 0.01
		self.conv1 = torch.nn.Sequential(
			#64x64x1
			nn.Conv2d(1,16,3,stride=2,padding=1,dilation=1),
			torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#32x32x16
			nn.Conv2d(16,32,3,stride=2,padding=2,dilation=2),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#32x32x16
			nn.Conv2d(32,64,3,stride=2,padding=3,dilation=3),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#64x8x8
			#8x8x64
			nn.ConvTranspose2d(64,32,4,stride=2,padding=1),
			torch.nn.LayerNorm((32, 16, 16)),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#16x16x32
			nn.ConvTranspose2d(32,32,4,stride=2,padding=1),
			torch.nn.LayerNorm((32, 32, 32)),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#16x32x32
			nn.ConvTranspose2d(32,16,4,stride=2,padding=1),
			torch.nn.LayerNorm((16, 64, 64)),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#4x64x64
			nn.ConvTranspose2d(16,8,4,stride=2,padding=1),
			torch.nn.LayerNorm((8, 128, 128)),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#2x128x128
			nn.ConvTranspose2d(8,4,3,stride=1,padding=1),
			torch.nn.LeakyReLU(0.2),
			#2x128x128
			nn.ConvTranspose2d(4,1,3,stride=1,padding=1),
			torch.nn.Sigmoid(),
			)

		self.conv2 = torch.nn.Sequential(
			#1x64x64
			nn.ConvTranspose2d(1,2,4,stride=2,padding=1),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			torch.nn.Sigmoid(),
			)


		self.conv_mixer = torch.nn.Sequential(
			#2x128x128
			nn.Conv2d(4,2,3,stride=1,padding=1),
			torch.nn.ReLU(),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			torch.nn.Sigmoid(),
			#1x128x128
			)

	def forward(self, x):
		bilinear = nn.functional.interpolate(x,scale_factor=2, mode='bilinear', align_corners = False)
		nearest = nn.functional.interpolate(x,scale_factor=2, mode='nearest')
		# x = self.conv0(x)
		conv2 = self.conv2(x)
		conv1 = self.conv1(x)
		# x = (conv1+1)/2
		x = torch.cat((bilinear,nearest,conv1,conv2),1)
		x = self.conv_mixer(x)
		return x

class Network2(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		
		p = 0.2
		self.conv1 = torch.nn.Sequential(
			#64x64x1
			nn.Conv2d(1,16,4,stride=2,padding=1),
			# torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#32x32x16
			nn.Conv2d(16,32,4,stride=2,padding=1),
			# torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
			#16x16x32
			nn.Conv2d(32,64,4,stride=2,padding=1),
			# torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(p),
		# 	#8x8x64
		# Deconv
		# 	#64x8x8
			nn.ConvTranspose2d(64,32,4,stride=2,padding=1),
			torch.nn.LayerNorm((32, 16, 16)),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#32x16x16
			nn.ConvTranspose2d(32,16,4,stride=2,padding=1),
			torch.nn.LayerNorm((16, 32, 32)),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#16x32x32
			nn.ConvTranspose2d(16,8,4,stride=2,padding=1),
			torch.nn.LayerNorm((8, 64, 64)),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#8x64x64
			nn.ConvTranspose2d(8,4,3,stride=1,padding=1),
			torch.nn.LayerNorm((4, 64, 64)),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#4x64x64
			nn.ConvTranspose2d(4,2,4,stride=2,padding=1),
			torch.nn.LayerNorm((2, 128, 128)),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#2x128x128
			nn.ConvTranspose2d(2,1,3,stride=1,padding=1),
			torch.nn.Dropout2d(p),
			torch.nn.Sigmoid(),
			)

		self.conv2 = torch.nn.Sequential(
			#1x64x64
			nn.ConvTranspose2d(1,2,4,stride=2,padding=1),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			torch.nn.Sigmoid(),
			)
				# self.conv3 = torch.nn.Sequential(
		# 	#2x128x128
		# 	nn.Conv2d(4,2,3,stride=1,padding=2,dilation=2),
		# 	torch.nn.Dropout2d(p),
		# 	torch.nn.ReLU(),
		# 	#2x128x128
		# 	nn.Conv2d(2,2,3,stride=1,padding=3,dilation=3),
		# 	torch.nn.Dropout2d(p),
		# 	torch.nn.ReLU(),
		# 	#2x128x128
		# 	nn.Conv2d(2,1,3,stride=1,padding=4,dilation=4),
		# 	torch.nn.Dropout2d(p),
		# 	torch.nn.ReLU(),
		# 	#2x128x128
		# 	nn.Conv2d(1,1,3,stride=1,padding=3,dilation=3),
		# 	torch.nn.Dropout2d(p),
		# 	torch.nn.ReLU(),
		# 	#2x128x128
		# 	nn.Conv2d(1,1,3,stride=1,padding=2,dilation=2),
		# 	torch.nn.Dropout2d(p),
		# 	torch.nn.ReLU(),
		# 	#2x128x128
		# 	nn.Conv2d(1,1,3,stride=1,padding=1,dilation=1),
		# 	torch.nn.Sigmoid(),
		# 	#1x128x128
		# 	)


		self.conv_mixer = torch.nn.Sequential(
			#2x128x128
			nn.Conv2d(4,2,3,stride=1,padding=1),
			torch.nn.ReLU(),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			torch.nn.Sigmoid(),
			#1x128x128
			)

	def forward(self, x):
		bilinear = nn.functional.interpolate(x,scale_factor=2, mode='bilinear', align_corners = False)
		nearest = nn.functional.interpolate(x,scale_factor=2, mode='nearest')
		# x = self.conv0(x)
		conv2 = self.conv2(x)
		conv1 = self.conv1(x)
		# x = (conv1+1)/2
		x = torch.cat((bilinear,nearest,conv1,conv2),1)
		x = self.conv_mixer(x)
		return x



class Network_res_128_2(nn.Module):
	def __init__(self):
		super(Network_res_128_2, self).__init__()
		p = 0.01
		self.conv1 = torch.nn.Sequential(
			#64x64x1
			nn.Conv2d(1,64,3,stride=1,padding=1,dilation=1),
			torch.nn.Dropout2d(p),
			torch.nn.ReLU(),
			nn.Conv2d(64,64,3,stride=1,padding=2,dilation=2),
			nn.BatchNorm2d(64),
			torch.nn.Dropout2d(p),
			torch.nn.ReLU(),
			nn.Conv2d(64,64,3,stride=1,padding=3,dilation=3),
			nn.BatchNorm2d(64),
			torch.nn.Dropout2d(p),
			torch.nn.ReLU(),
			nn.Conv2d(64,64,3,stride=1,padding=4,dilation=4),
			nn.BatchNorm2d(64),
			torch.nn.Dropout2d(p),
			torch.nn.ReLU(),
			nn.Conv2d(64,64,3,stride=1,padding=3,dilation=3),
			nn.BatchNorm2d(64),
			torch.nn.Dropout2d(p),
			torch.nn.ReLU(),
			nn.Conv2d(64,64,3,stride=1,padding=2,dilation=2),
			nn.BatchNorm2d(64),
			torch.nn.Dropout2d(p),
			torch.nn.ReLU(),
			nn.Conv2d(64,1,3,stride=1,padding=1,dilation=1),
			)

		self.conv2 = torch.nn.Sequential(
			#1x64x64
			nn.Conv2d(1,2,3,stride=1,padding=1),
			torch.nn.Dropout2d(p),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			# torch.nn.Sigmoid(),
			)


		self.conv_mixer = torch.nn.Sequential(
			#2x128x128
			nn.Conv2d(3,2,3,stride=1,padding=1),
			torch.nn.Dropout2d(p),
			torch.nn.ReLU(),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			# torch.nn.Sigmoid(),
			#1x128x128
			)

	def forward(self, x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(x)
		# x = (conv1+1)/2
		x = torch.cat((x,conv1,conv2),1)
		x = self.conv_mixer(x)
		return x

class Network_res_(nn.Module):
	def __init__(self):
		super(Network_res_, self).__init__()
		p = 0.2
		self.conv1 = torch.nn.Sequential(
			#64x64x1
			nn.Conv2d(1,64,3,stride=1,padding=1),
			torch.nn.ReLU(),
			nn.Conv2d(64,64,3,stride=1,padding=1),
			nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			nn.Conv2d(64,64,3,stride=1,padding=1),
			nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			nn.Conv2d(64,64,3,stride=1,padding=1),
			nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			nn.Conv2d(64,1,3,stride=1,padding=1),
			)

		self.conv2 = torch.nn.Sequential(
			#1x64x64
			nn.Conv2d(1,2,3,stride=1,padding=1),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			# torch.nn.Sigmoid(),
			)


		self.conv_mixer = torch.nn.Sequential(
			#2x128x128
			nn.Conv2d(3,2,3,stride=1,padding=1),
			torch.nn.ReLU(),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			# torch.nn.Sigmoid(),
			#1x128x128
			)

	def forward(self, x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(x)
		# x = (conv1+1)/2
		x = torch.cat((x,conv1,conv2),1)
		x = self.conv_mixer(x)
		return x

class Network_res_(nn.Module):
	def __init__(self):
		super(Network_res_, self).__init__()
		p = 0.2
		self.conv1 = torch.nn.Sequential(
			#64x64x1
			nn.Conv2d(1,64,3,stride=1,padding=1),
			torch.nn.ReLU(),
			nn.Conv2d(64,64,3,stride=1,padding=1),
			nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			nn.Conv2d(64,64,3,stride=1,padding=1),
			nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			nn.Conv2d(64,64,3,stride=1,padding=1),
			nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			nn.Conv2d(64,1,3,stride=1,padding=1),
			)

		self.conv2 = torch.nn.Sequential(
			#1x64x64
			nn.Conv2d(1,2,3,stride=1,padding=1),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			# torch.nn.Sigmoid(),
			)


		self.conv_mixer = torch.nn.Sequential(
			#2x128x128
			nn.Conv2d(3,2,3,stride=1,padding=1),
			torch.nn.ReLU(),
			#2x128x128
			nn.Conv2d(2,1,3,stride=1,padding=1),
			# torch.nn.Sigmoid(),
			#1x128x128
			)

	def forward(self, x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(x)
		# x = (conv1+1)/2
		x = torch.cat((x,conv1,conv2),1)
		x = self.conv_mixer(x)
		return x
