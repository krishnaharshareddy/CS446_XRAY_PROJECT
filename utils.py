import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim 
from torch.autograd import Variable
from utils import *
import pdb
def tv_loss(yhat, y, norm=1):
    bsize, chan, height, width = y.size()
    dy = torch.abs(y[:,:,1:,:] - y[:,:,:-1,:])
    dyhat = torch.abs(yhat[:,:,1:,:] - yhat[:,:,:-1,:])
    error = torch.norm(dy - dyhat, norm)/(height*width*bsize)
    dx = torch.abs(y[:,:,:,1:] - y[:,:,:,:-1])
    dxhat = torch.abs(yhat[:,:,:,1:] - yhat[:,:,:,:-1])
    error = error + torch.norm(dx - dxhat, norm)/(height*width*bsize)
    return error 
def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)], device=torch.device("cuda"))
    return gauss/gauss.sum()
def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 4).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window.cuda()
def ssim(x, x_hat, window_size=8):
    window = create_window(window_size, x.size(1))
    return _ssim(x, x_hat, window, window_size, x.size(1))