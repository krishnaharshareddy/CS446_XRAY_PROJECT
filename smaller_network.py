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
class Network_up(nn.Module):
	def __init__(self):
		super(Network_up, self).__init__()
		
		p = 0.01
		self.conv1 = torch.nn.Sequential(
			#64x64x1
			nn.BatchNorm2d(2),
			nn.Conv2d(2,16,3,stride=2,padding=1,dilation=1, bias=False),
			torch.nn.BatchNorm2d(16),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#16x16x16
			nn.Conv2d(16,32,3,stride=2,padding=2,dilation=2, bias=False),
			torch.nn.BatchNorm2d(32),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#32x32x16
			nn.Conv2d(32,64,3,stride=2,padding=3,dilation=3, bias=False),
			torch.nn.BatchNorm2d(64),
			torch.nn.LeakyReLU(0.2),
			torch.nn.Dropout2d(p),
			#64x8x8
			#8x8x64
			nn.ConvTranspose2d(64,32,4,stride=2,padding=1, bias=False),
			torch.nn.BatchNorm2d(32),
			torch.nn.LeakyReLU(0.20),
			torch.nn.Dropout2d(p),
			#16x16x32
			nn.ConvTranspose2d(32,32,4,stride=2,padding=1, bias=False),
			torch.nn.BatchNorm2d(32),
			torch.nn.LeakyReLU(0.20),
			torch.nn.Dropout2d(p),
			#16x32x32
			nn.ConvTranspose2d(32,16,4,stride=2,padding=1, bias=False),
			torch.nn.BatchNorm2d(16),
			torch.nn.LeakyReLU(0.20),
			torch.nn.Dropout2d(p),
			#4x64x64
			nn.ConvTranspose2d(16,8,4,stride=2,padding=1, bias=False),
			torch.nn.BatchNorm2d(8),
			torch.nn.LeakyReLU(0.20),
			torch.nn.Dropout2d(p),
			#2x128x128
			nn.ConvTranspose2d(8,4,3,stride=1,padding=1, bias=False),
			torch.nn.LeakyReLU(0.2),
			#2x128x128
			nn.ConvTranspose2d(4,1,3,stride=1,padding=1, bias=False),
			torch.nn.Sigmoid(),
			)

		self.conv2 = torch.nn.Sequential(
			#1x64x64
			nn.ConvTranspose2d(2,32,4,stride=2,padding=1, bias=False),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			#2x128x128
			nn.Conv2d(32,1,3,stride=1,padding=1, bias=False),
			torch.nn.Sigmoid(),
			)


	def forward(self, x):
		conv2 = self.conv2(x)
		conv1 = self.conv1(x)
		bilinear = nn.functional.interpolate(x[:,0,:,:].unsqueeze(1),scale_factor=2, mode='bilinear', align_corners = False)
		# nearest = nn.functional.interpolate(x,scale_factor=2, mode='nearest')
		# x = self.conv0(x)
		# x = (conv1+1)/2
		# x = torch.cat((bilinear,nearest,conv1,conv2),1)
		x = conv1+conv2+bilinear
		return x


class Network_res_128_(nn.Module):
	def __init__(self):
		super(Network_res_128_, self).__init__()
		p = 0.02
		self.conv1 = torch.nn.Sequential(
			#64x64x1
			nn.BatchNorm2d(1),
			nn.Conv2d(1,64,3,stride=1,padding=1,dilation=1, bias=False),
			nn.BatchNorm2d(64),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			nn.Conv2d(64,64,3,stride=1,padding=2,dilation=2, bias=False),
			nn.BatchNorm2d(64),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			nn.Conv2d(64,64,3,stride=1,padding=3,dilation=3, bias=False),
			nn.BatchNorm2d(64),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			nn.Conv2d(64,64,3,stride=1,padding=2,dilation=2, bias=False),
			nn.BatchNorm2d(64),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			nn.Conv2d(64,1,3,stride=1,padding=1,dilation=1, bias=False),
			)

	def forward(self, x):
		x = self.conv1(x)
		return x


class Network_res_(nn.Module):
	def __init__(self):
		super(Network_res_, self).__init__()
		p = 0.02
		self.conv1 = torch.nn.Sequential(
			#64x64x1
			nn.Conv2d(1,64,3,stride=1,padding=1, dilation = 1, bias=False),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			nn.Conv2d(64,64,3,stride=1,padding=2, dilation = 2, bias=False),
			nn.BatchNorm2d(64),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			nn.Conv2d(64,64,3,stride=1,padding=3, dilation = 3, bias=False),
			nn.BatchNorm2d(64),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			nn.Conv2d(64,64,3,stride=1,padding=2, dilation = 2, bias=False),
			nn.BatchNorm2d(64),
			torch.nn.Dropout2d(p),
			torch.nn.LeakyReLU(0.2),
			nn.Conv2d(64,1,3,stride=1,padding=1, dilation = 1, bias=False),
			)

	def forward(self, x):
		x = self.conv1(x)
		return x
