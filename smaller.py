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
from utils import ssim,tv_loss
import torchvision.transforms.functional as TF
import torchvision
import numpy.random as random
from tensorboardX import SummaryWriter
from utils import LossHandler
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

def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch or epoch>35:
        return optimizer
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer

from smaller_network import  Network_res_, Network_up, Network_res_128_

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
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--lr', type=float, default=5e-3)
	parser.add_argument('--l1', type=float, default=1e1)
	parser.add_argument('--num_iters', type=int, default=100)
	parser.add_argument("--train", dest="train", default=False, action="store_true")  # noqa
	parser.add_argument('--test_save_path', default='test')
	parser.add_argument('--load_earlier', dest="load_earlier", default=False, action="store_true")
	args = parser.parse_args()

	train_loader,val_loader,test_loader = read_data(args.batch_size)
	network = Network_res_()
	network.apply(weights_init)
	network_up = Network_up()
	network_up.apply(weights_init)
	network_128 = Network_res_128_()
	network_128.apply(weights_init)
	
	print(network)
	train_loss = []
	val_loss = []
	model_dir = 'models/'+args.model_name
	tensorboard_dir = model_dir + '/tensorboard'
	mkdir_p(model_dir)
	if args.train:
		writer_tf = SummaryWriter(log_dir=tensorboard_dir)
		loss_handler = LossHandler()
		optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
		optimizer_up = torch.optim.Adam(network_up.parameters(), lr=args.lr)
		optimizer_128 = torch.optim.Adam(network_128.parameters(), lr=args.lr)
		network.cuda()
		network_up.cuda()
		network_128.cuda()
		for epoch in range(args.num_iters):
			i = epoch
			network.train()
			network_up.train()
			network_128.train()
			for idx, x in enumerate(train_loader):
				img64 = x['img64'].cuda()
				img128 = x['img128'].cuda()
				imgname = x['img_name']
				optimizer_128.zero_grad()
				optimizer_up.zero_grad()
				optimizer.zero_grad()
				g_img64_res = network(img64)
				g_img64 = img64-g_img64_res
				loss = ((255*(img64-g_img64))**2).mean()/100
				g_img128 = network_up(torch.cat((g_img64,img64),1))
				g_img128_res = network_128(g_img128)
				ssim_loss = ssim(g_img128,img128)
				loss -=  args.l1*ssim_loss
				g_img128_denoised = g_img128-g_img128_res
				l2_loss = ((255*(g_img128_denoised-img128))**2).mean()
				l1_loss = (abs(255*(g_img128_denoised-img128))).mean()
				rmse_loss = rmse(g_img128_denoised,img128)
				ssim_loss = ssim(g_img128_denoised,img128)
				loss += l2_loss  
				loss.backward()
				optimizer_128.step()
				optimizer_up.step()
				optimizer.step()


				if idx%10 ==0:
					print("{} TRAINING {} {}: RMSE_LOSS:{} SSIM:{} L1:{} tv:{} TOTAL:{} ".format(args.model_name,
						epoch,idx,
						(rmse_loss.detach().cpu().numpy()),
						ssim_loss.detach().cpu().numpy(),
						l1_loss.detach().cpu().numpy(),
						0,#tv_losss.detach().cpu().numpy(),
						loss.detach().cpu().numpy()))
				
			train_loss.append((rmse(img128,g_img128_denoised).detach().cpu().numpy()))

			loss_handler['loss'][i].append(loss.item())
			loss_handler['rmse_loss'][i].append(rmse_loss.item())
			loss_handler['ssim_loss'][i].append(ssim_loss.item())
			loss_handler['l2_loss'][i].append(l2_loss.item())
			loss_handler['l1_loss'][i].append(l1_loss.item())
			with torch.no_grad(): 
				loss_sum = 0.0
				network.eval()
				network_up.eval()
				network_128.eval()
				for idx,x in enumerate(val_loader):
					img64 = x['img64'].cuda()
					img128 = x['img128'].cuda()
					imgname = x['img_name']
					g_img64_res = network(img64)
					g_img64 = img64-g_img64_res
					g_img128 = network_up(torch.cat((g_img64,img64),1))
					g_img128_res = network_128(g_img128)
					g_img128_denoised = g_img128-g_img128_res
					loss2 = (rmse(img128,g_img128_denoised).detach().cpu().numpy())

					if idx%10 ==0:
						print("EVAL: RMSE_LOSS:{} ".format(loss2))
					loss_sum += loss2
				val_loss.append(loss_sum/(idx+1))

			loss_handler['val_loss'][i].append(loss_sum/(idx+1))
			loss_handler.log_epoch(writer_tf, i)
			# mkdir_p('./models/')
				
			if epoch%2 == 0:
				torch.save(network, '{}/{}_{}.pt'.format(model_dir, args.model_name, str(epoch)))
				torch.save(network_up, '{}/{}_{}_up.pt'.format(model_dir, args.model_name, str(epoch)))
				torch.save(network_128, '{}/{}_{}_128.pt'.format(model_dir, args.model_name, str(epoch)))
			# optimizer = exp_lr_scheduler(optimizer, epoch)

	else:
		# Load a pretrained model and use that to make the final images
		network = torch.load((args.model_name))
		network_128 = torch.load((args.model_name))
		network_up = torch.load((args.model_name))
		network.eval()
		network_up.eval()
		network_128.eval()
		for idx, x in enumerate(test_loader):
			img64 = x['img64'].cuda()
			imgname = x['img_name']
			g_img64_res = network(img64)
			g_img64 = img64-g_img64_res
			g_img128 = network_up(torch.cat((g_img64,img64),1))
			g_img128_res = network_128(g_img128)
			g_img128_denoised = g_img128-g_img128_res
			save_images('./images/'+args.model_name,imgname,g_img128_denoised)
	pdb.set_trace()


