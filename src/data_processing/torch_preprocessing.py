import torch
import torchvision.transforms.functional as F
import numpy as np


class Identity():

    def __call__(self, x): return x

    def train(self, x): return x


class MaskedNormalize(object):

    """Normalize a tensor image with mean and standard deviation, with a mask.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean, std, background=None, inplace=False):
        self.mean = mean
        self.std = std
        self.bg = background
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        if self.bg is None:
            return F.normalize(tensor, self.mean, self.std, self.inplace)

        # tensor_mask = tensor + 0
        res = tensor + 0
        # cs, xs, ys = np.where(tensor != self.bg)
        # stds = tensor.std((1, 2))
        # for channel in np.unique(cs):
        #     xs_fg = xs[cs == channel]
        #     ys_fg = ys[cs == channel]
        #     fg = tensor[channel, xs_fg, ys_fg]
        #     tensor_mask[channel][tensor[channel] == self.bg] = fg.mean()
        #     stds[channel] = self.std[channel] * len((tensor == self.bg)[channel]) / len(fg)

        normed = F.normalize(tensor, self.mean, self.std)
        res[tensor != self.bg] = normed[tensor != self.bg]
        return res

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MaskedMinMaxNorm(object):

    iindex = 0

    def __init__(self, background_prev=None, background_next=None):
        self.bg1 = background_prev
        self.bg2 = background_next

    def __call__(self, tensor):
        if self.bg1 is None:
            return self.min_max_norm(tensor)
        res = tensor + 0
        if tensor.squeeze().ndim == 3:
            for chan in range(tensor.shape[0]):
                fg = tensor[chan][tensor[chan] != self.bg1]
                if len(fg.unique()) == 1:
                    res[chan] = 1
                else:
                    # from src.plotter import create_pillow_image
                    # import pathlib
                    # pathlib.Path('./some_imgs').mkdir(exist_ok=True, parents=True)
                    # pil_to_save = create_pillow_image([[res[0].cpu().numpy(), res[1].cpu().numpy(), res[2].cpu().numpy()]])

                    # pil_to_save.save('./some_imgs/bg{}.png'.format(self.iindex))
                    # self.iindex+=1
                    # print('saved......')
                    # print('bg:', (res[chan] != self.bg1).sum())
                    # print(res[chan].shape)
                    # print(res.shape)
                    res[chan][res[chan] != self.bg1] = self.min_max_norm(fg)
        else:
            fg = tensor[tensor != self.bg1]
            res[res != self.bg1] = self.min_max_norm(fg)
        res[tensor == self.bg1] = self.bg2
        return res

    def min_max_norm(self, tensor):

        if tensor.squeeze().ndim == 3:
            mins = torch.zeros(tensor.shape[0]).to(tensor.device)
            maxs = torch.zeros(tensor.shape[0]).to(tensor.device)
            for chan in range(tensor.shape[0]):
                mins[chan], _ = tensor[chan].min()
                maxs[chan], _ = tensor[chan].max()


        else:
            mins = tensor.min()
            maxs = tensor.max()
        return tensor.sub_(mins).div_(maxs - mins)

    def __repr__(self):
        return self.__class__.__name__ + '(bg_prev={0}, bg_next={1})'.format(self.bg1, self.bg2)


class ToDevice(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, tensor):
        return tensor.to(self.device)

    def __repr__(self):
        return self.__class__.__name__ + '(device={})'.format(self.device)


class ToUint8(object):

    def __init__(self, bg=None):
        self.bg = bg

    def __call__(self, img):
        if self.bg is None:
            img = (img - img.min((1, 2))[:, None, None]) / (img.max((1, 2))[:, None, None] - img.min((1, 2))[:, None, None])
            return (img * 255).astype(np.uint8)
        mask = img == self.bg
        curmin = np.where(mask, img.max(), img).min((1, 2))
        # img[~mask].min((1, 2))
        res = np.where(mask, curmin[:, None, None], img)
        res = (res - curmin[:, None, None]) / (res.max((1, 2))[:, None, None] - res.min((1, 2))[:, None, None])
        return np.where(mask, 0, res*254+1).astype(np.uint8)

    def __repr__(self):
        return self.__class__.__name__
