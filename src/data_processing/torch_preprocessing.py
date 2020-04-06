import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

class ToDevice(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, tensor):
        return tensor.to(self.device)

    def __repr__(self):
        return self.__class__.__name__ + '(device={})'.format(self.device)

class ToUint8(object):

    def __call__(self, img):
        img = (img - img.min((1, 2))[:, None, None]) / (img.max((1, 2))[:, None, None] - img.min((1, 2))[:, None, None])
        return (img * 255).astype(np.uint8)

    def __repr__(self):
        return self.__class__.__name__ 

    