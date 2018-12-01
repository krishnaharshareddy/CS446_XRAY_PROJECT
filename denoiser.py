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
def mkdir_p(path):
    try:
        os.makedirs(path)
    except:
    	pass

def pil_loader(path):
    img = Image.open(path)
    return img

class xrayDataset(torch.utils.data.Dataset):
	def __init__(self, root='../', transform=None, train=True):
		"""
		Args:
		    root_dir (string): Directory with all the images.
		    transform (callable, optional): Optional transform to be applied
		        on a sample.
		"""
		self.root_dir = root
		self.transform = transform
		self.train = train

	def __len__(self):
		return 16000 if self.train else 3999

	def __getitem__(self, idx):

		if self.train:
			s = str(int(idx)+4000).zfill(5)
			img_name = os.path.join(self.root_dir,
			                        'train_images_128x128/train_{}.png'.format(s))
			image = pil_loader(img_name)
			img_name = os.path.join(self.root_dir,
			                        'train_images_64x64/train_{}.png'.format(s))
			image64 = pil_loader(img_name)
			if self.transform:
				if random.random() > 0.5:
					image = TF.hflip(image)
					image64 = TF.hflip(image64)
				if random.random() > 0.5:
					image = TF.vflip(image)
					image64 = TF.vflip(image64)
				image = self.transform(image)
				image64 = self.transform(image64)# Random horizontal flipping
			sample = {'img128': image, 'img64': image64, 'img_name':s}

		else:
			s = str(int(idx)+1).zfill(5)
			img_name = os.path.join(self.root_dir,
			                        'test_images_64x64/test_{}.png'.format(s))
			image64 = pil_loader(img_name)
			transform = transforms.Compose([
					transforms.Grayscale(),
					transforms.ToTensor()
			])
			if self.transform:
				image64 = transform(image64)
			sample = {'img64': image64, 'img_name':s}

		return sample

def read_data(batch_size,split=0.2):
    print('Loading data...')
    curr = time.time()
    transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
    dataset = xrayDataset(root='../',transform=transform,train=True)

    train_dataset,val_dataset = torch.utils.data.dataset.random_split(dataset,[16000-int(16000*(split)),int(16000*(split))])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataset = xrayDataset(root='../',transform=transform,train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print('Time spent on loading: {}, now proceeding to training...'.format(time.time()-curr))
    return train_loader,val_loader,test_loader

class Network(nn.Module):
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
def weights_init(m):
	if isinstance(m, torch.nn.Conv2d):
		torch.nn.init.xavier_uniform_(m.weight.data)
	if isinstance(m, torch.nn.ConvTranspose2d):
		torch.nn.init.xavier_uniform_(m.weight.data)

def save_images(root,imagenames,images):
	mkdir_p(root)
	for i,name in enumerate(imagenames):
		torchvision.utils.save_image(images[i],root+'/test_{}.png'.format(name))

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.manual_seed(3)
def rmse(img1,img2):
	x = ((img1-img2)**2).sum([1,2,3])
	x = torch.sqrt(x).mean()
	return x
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', default='test')
	parser.add_argument('--batch_size', default=64)
	parser.add_argument('--lr', default=5e-3)
	parser.add_argument('--l1', default=1e-1)
	parser.add_argument('--num_iters', default=100)
	parser.add_argument("--train", dest="train", default=False, action="store_true")  # noqa
	parser.add_argument('--test_save_path', default='test')
	args = parser.parse_args()

	train_loader,val_loader,test_loader = read_data(args.batch_size)
	network = Network()
	network.apply(weights_init)
	print(network)
	train_loss = []
	val_loss = []
	if args.train:
		optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
		network.cuda()
		for epoch in range(args.num_iters):
			train_loader,val_loader,test_loader = read_data(args.batch_size)
			network.train()
			for idx, x in enumerate(train_loader):
				img64 = x['img64'].cuda()
				img128 = x['img128'].cuda()
				imgname = x['img_name']

				optimizer.zero_grad()

				g_img128 = network(img64)

				l2_loss = ((img128-g_img128)**2).mean()
				l1_loss = (abs(img128-g_img128)).mean()
				rmse_loss = rmse(img128,g_img128)
				ssim_loss = ssim(img128,g_img128)
				loss = rmse_loss #+ l1_loss - args.l1*ssim_loss
				loss.backward()
				optimizer.step()
				if idx%10 ==0:
					print("TRAINING {} {}: RMSE_LOSS:{} SSIM:{} L1:{} L2:{} TOTAL:{} ".format(epoch,idx,
						(rmse(img128,g_img128).detach().cpu().numpy()),
						ssim_loss.detach().cpu().numpy(),
						l1_loss.detach().cpu().numpy(),
						l2_loss.detach().cpu().numpy(),
						loss.detach().cpu().numpy()))
			train_loss.append((rmse(img128,g_img128).detach().cpu().numpy()))

			loss_sum = 0.0
			network.eval()
			for idx,x in enumerate(val_loader):
				img64 = x['img64'].cuda()
				img128 = x['img128'].cuda()
				imgname = x['img_name']
				g_img128 = network(img64)
				loss = (rmse(img128,g_img128).detach().cpu().numpy())
				if idx%10 ==0:
					print("EVAL: RMSE_LOSS:{} ".format(loss))
				loss_sum += loss
			val_loss.append(loss_sum/(idx+1))
			mkdir_p('./models/')
			torch.save(network, './models/{}_{}.pt'.format(args.model_name, str(epoch)))
	else:
		network.eval()
		# Load a pretrained model and use that to make the final images
		network = torch.load('./models/{}.pt'.format(args.model_name))
		for idx, x in enumerate(test_loader):
			img64 = x['img64'].cuda()
			imgname = x['img_name']
			g_img128 = network(img64)
			save_images('./images/'+args.model_name,imgname,g_img128)
	pdb.set_trace()


