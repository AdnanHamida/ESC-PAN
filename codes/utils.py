import torch
from torch import nn
import numpy as np
from math import log10, sqrt
from tqdm.auto import trange, tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def mod_crop2(image, scale):
    h, w = image.shape[2], image.shape[3]
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    return (image[:,:,:h, :w])
def psnr(loss):
  return 20 * log10(1 / sqrt(loss))
def Clip_func(input):

  """ Saturates any value that is above or below 0 and 1 """
  return torch.clamp(input,min=0.0 ,max=1.0)


class Clip(nn.Module):
  """ Activation function that saturates any value that is above or below 0 and 1 """
  def __init__(self):
    super().__init__()

  def forward(self, input):
    return Clip_func(input)
def rescaling(x, feature_range=(0, 1)):
    a, b =feature_range
    x=a+(b-a)*(x-x.min())/(x.max()-x.min())
    return x
