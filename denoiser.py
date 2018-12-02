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

from network import Network
from network import Discriminator

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
	x = ((255*(img1-img2))**2).mean(1).mean(1).mean(1)
	x = torch.sqrt(x).mean()
	return x
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', default='test')
	parser.add_argument('--batch_size', default=64)
	parser.add_argument('--lr', type=float, default=5e-3)
	parser.add_argument('--l1', type=float, default=1e1)
	parser.add_argument('--num_iters', default=100)
	parser.add_argument("--train", dest="train", default=False, action="store_true")  # noqa
	parser.add_argument('--test_save_path', default='test')
	args = parser.parse_args()

	train_loader,val_loader,test_loader = read_data(args.batch_size)
	network = Network()
	network.apply(weights_init)

	d = Discriminator()
	d.apply(weights_init)
	bce_loss = torch.nn.BCEWithLogitsLoss()
	true_crit, false_crit = torch.ones(args.batch_size, 1, device='cuda'), torch.zeros(args.batch_size, 1, device='cuda')
	print(network)
	train_loss = []
	val_loss = []
	if args.train:
		optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
		d_optimizer = torch.optim.Adam(d.parameters(), lr=args.lr)
		network.cuda()
		d.cuda()
		for epoch in range(args.num_iters):
			train_loader,val_loader,test_loader = read_data(args.batch_size)
			network.train()
			for idx, x in enumerate(train_loader):
				img64 = x['img64'].cuda()
				img128 = x['img128'].cuda()
				imgname = x['img_name']
				optimizer.zero_grad()
				g_img128 = network(img64)
				l2_loss = ((255*(img128-g_img128))**2).mean()
				l1_loss = (abs(255*(img128-g_img128))).mean()
				rmse_loss = rmse(img128,g_img128)
				ssim_loss = ssim(img128,g_img128)
				# dloss = bce_loss(d(g_img128,img64),true_crit)
				loss = l2_loss + l1_loss  - args.l1*ssim_loss
				loss.backward()
				optimizer.step()

				# d_optimizer.zero_grad()
				# g_img128 = network(img64)
				# g_img128.detach()
				# dloss = bce_loss(d(img128,img64),true_crit)+bce_loss(d(g_img128,img64),false_crit)
				# dloss.backward()
				# d_optimizer.step()


				if idx%10 ==0:
					print("TRAINING {} {}: RMSE_LOSS:{} SSIM:{} L1:{} d:{} TOTAL:{} ".format(epoch,idx,
						(rmse(img128,g_img128).detach().cpu().numpy()),
						ssim_loss.detach().cpu().numpy(),
						l1_loss.detach().cpu().numpy(),
						0,#dloss.detach().cpu().numpy(),
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


