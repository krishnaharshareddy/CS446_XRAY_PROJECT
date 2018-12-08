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
from utils import ssim,tv_loss,msssim
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



from network import Network, Network_res_, Network_up, Network_res_128_2
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
	network_128 = Network_res_128_2()
	network_128.apply(weights_init)

	bce_loss = torch.nn.BCEWithLogitsLoss()
	true_crit, fake_crit = torch.ones(args.batch_size, 1, device='cuda'), torch.zeros(args.batch_size, 1, device='cuda')
	
	print(network)
	train_loss = []
	val_loss = []
	if args.train:
		optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
		optimizer_up = torch.optim.Adam(network_up.parameters(), lr=args.lr)
		optimizer_128 = torch.optim.Adam(network_128.parameters(), lr=args.lr)
		if args.load_earlier:
			network = torch.load('./models/{}.pt'.format('residual_denoiser_4_199'))
			# network_128 = torch.load('./models/{}_128.pt'.format('residual_denoiser_4_199'))
			network_up = torch.load('./models/{}_up.pt'.format('residual_denoiser_4_199'))
			# network = torch.load('./models/{}.pt'.format('residual_denoiser_5_66'))
			# network_128 = torch.load('./models/{}_128.pt'.format('residual_denoiser_5_66'))
			# network_up = torch.load('./models/{}_up.pt'.format('residual_denoiser_5_66'))
			pass

		network.cuda()
		network_up.cuda()
		network_128.cuda()
		for epoch in range(args.num_iters):
#			train_loader,val_loader,test_loader = read_data(args.batch_size)
			network.train()
			network_up.train()
			network_128.train()
			for idx, x in enumerate(train_loader):
				img64 = x['img64'].cuda()
				img128 = x['img128'].cuda()
				imgname = x['img_name']

#[24.192374267578124, 13.780136795043946, 10.337756748199462, 9.317650623321533, 12.128365478515626, 7.850205125808716, 7.567445421218872, 6.8736280250549315, 6.615870399475098, 6.383710336685181, 6.231422824859619, 6.342484464645386, 6.041673736572266, 6.490949573516846, 5.888002920150757, 5.85293173789978, 5.831213283538818, 5.762629537582398, 5.792541933059693, 6.49580602645874, 5.710175485610962, 5.708357334136963, 5.624448575973511, 5.64095947265625, 5.534252462387085, 6.518613176345825, 5.504629144668579, 5.485298805236816, 5.5180031681060795, 5.482980947494507, 5.503473739624024, 5.430056095123291, 5.418265590667724, 5.454664716720581, 5.468684349060059, 5.372878246307373, 5.533935070037842, 5.463591012954712, 5.298991832733154, 5.478468074798584, 5.324517030715942, 5.377024936676025, 5.398367881774902, 5.359409818649292, 5.243158760070801, 5.413276109695435, 5.424651193618774, 5.251041126251221, 5.283333158493042, 5.34383900642395, 5.243055419921875, 5.295056886672974, 5.2778957748413085, 5.316831569671631, 5.170322256088257, 5.203807487487793, 5.207704858779907, 5.226997661590576, 5.357512092590332, 5.23078239440918, 5.1377192401885985, 5.2168605995178225, 5.172273302078247, 5.172620220184326, 5.226056280136109, 5.179642543792725, 5.195523128509522, 5.295998268127441, 5.1643625831604005, 5.382639398574829, 5.155825777053833, 5.320999937057495, 5.142209186553955, 5.213652868270874, 5.237732105255127, 5.142346868515014, 5.081751985549927, 5.095134668350219, 5.272763986587524, 5.105094366073608, 5.074364156723022, 5.1409611320495605, 5.192793531417847, 5.0785156345367435, 5.161319274902343, 5.0744526576995845, 5.1645317554473875, 5.066127424240112, 5.061043453216553, 5.134311170578003, 5.119212636947632, 5.094606542587281, 5.0712425327301025, 5.164579048156738, 5.099315328598022, 5.062582836151123, 5.1531647682189945, 5.027270193099976, 5.160202045440673, 5.092353210449219, 5.033833799362182, 5.026013631820678, 5.0779236316680905, 5.060612182617188, 5.138874406814575, 5.015364332199097, 5.130121879577636, 5.022044095993042, 5.02179741859436, 5.0663185310363765, 5.041223583221435, 5.118391199111938, 5.095140523910523, 5.089156370162964, 5.017580099105835, 5.019016485214234, 5.008958806991577, 4.984974975585938, 5.039338111877441, 4.986078948974609, 5.000931653976441, 5.001703882217408, 5.046366176605225, 5.119622259140015, 4.978967914581299, 5.026183738708496, 4.948789749145508, 4.941891670227051, 4.9761668395996095, 4.924230461120605, 5.018248462677002, 4.915916042327881, 4.890613775253296, 4.901108703613281, 4.863761882781983, 4.8825654125213624, 4.973881778717041, 4.946408367156982, 4.8392732715606686, 4.867066612243653, 4.827979030609131, 4.947094039916992, 4.807460231781006, 4.820450353622436, 4.828524103164673, 4.776957321166992, 4.8104703330993654, 4.834663524627685, 4.774981079101562, 4.78479344367981, 4.772664184570313, 4.785500621795654, 4.7768166065216064, 4.778760042190552, 4.793102350234985, 4.768957071304321, 4.829016189575196, 4.753011903762817, 4.749005155563355, 4.741369819641113, 4.752405891418457, 4.757156343460083, 4.741995763778687, 4.806871614456177, 4.732878541946411, 4.728358240127563, 4.770045232772827, 4.727316303253174, 4.756327104568482, 4.736267995834351, 4.74666543006897, 4.741891326904297, 4.745450868606567, 4.722893915176392, 4.8793871784210205, 4.731589736938477, 4.752817811965943, 4.764658241271973, 4.711772165298462, 4.743170881271363, 4.7795631694793705, 4.794848051071167, 4.799378490447998, 4.724600067138672, 4.7405033206939695, 4.877845287322998, 4.716140851974488, 4.71772723197937, 4.774738245010376, 4.734478998184204, 4.773439693450928, 4.70996166229248, 4.729413585662842, 4.72668155670166, 4.71386308670044, 4.701945867538452, 4.742524881362915, 4.7472872638702395, 4.703505058288574, 4.6835482215881346]


				# Learn to denoise 128x128
				optimizer_128.zero_grad()
				optimizer_up.zero_grad()
				optimizer.zero_grad()
				g_img64_res = network(img64)
				g_img64 = img64-g_img64_res
				# g_img64.detach()
				g_img128 = network_up(torch.cat((g_img64,img64),1))
				# g_img128.detach()
				g_img128_res = network_128(g_img128)
				g_img128_denoised = g_img128-g_img128_res
				l2_loss = ((255*(g_img128_denoised-img128))**2).mean()
				l1_loss = (abs(255*(g_img128_denoised-img128))).mean()
				rmse_loss = rmse(g_img128_denoised,img128)
				ssim_loss = ssim(g_img128_denoised,img128)
				loss = l2_loss  + l1_loss - args.l1*ssim_loss
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
				mkdir_p('./models/')
				
			if epoch%2 == 0:
				torch.save(network, './models/{}_{}.pt'.format(args.model_name, str(epoch)))
				torch.save(network_up, './models/{}_{}_up.pt'.format(args.model_name, str(epoch)))
				torch.save(network_128, './models/{}_{}_128.pt'.format(args.model_name, str(epoch)))
			# optimizer = exp_lr_scheduler(optimizer, epoch)

	else:
		# Load a pretrained model and use that to make the final images
		network = torch.load('./models/{}.pt'.format(args.model_name))
		network_128 = torch.load('./models/{}_128.pt'.format(args.model_name))
		network_up = torch.load('./models/{}_up.pt'.format(args.model_name))
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


