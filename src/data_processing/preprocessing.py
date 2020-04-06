import collections
import random

import cv2
import numpy as np
from skimage.morphology import disk, white_tophat
from skimage.filters import median
from skimage.exposure import equalize_adapthist, equalize_hist 
from scipy import ndimage

from src.data_processing.processing import ProcessImage, GlobalMorphProcess, ProcessorRow

Iterable = collections.abc.Iterable


class ToUint8(ProcessImage):

    def apply_to_img2d(self, img_orig, *args, **kwargs):
        img = (img_orig - img_orig.min()) / (img_orig.max() - img_orig.min())
        return (img * 255).astype(np.uint8)

    def apply_to_target(self, target, *args, **kwargs):
        return target.astype(np.uint8)
 


class MinMaxNorm(ProcessImage):

    def apply_to_img2d(self, img, *args, **kwargs):
        return (img - img.min()) / (img.max() - img.min())
        


class NormalizeImg(ProcessImage):

    def apply_to_img2d(self, img, *args, **kwargs):
        nonzero = img[img != 0]
        return (img - nonzero.mean()) / nonzero.std()


class Resize(ProcessImage):
    """Resize the input nd array Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(
        self, 
        size, 
        interpolation_input=cv2.INTER_LINEAR, 
        interpolation_target=cv2.INTER_NEAREST, 
        apply_to_target=True,
    ):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if type(size) in [int, np.int64]:
            size = (size, size)
        self.size = size
        self.interpolation_input = interpolation_input
        self.interpolation_target = interpolation_target
        self.apply_to_target = apply_to_target

    def apply_to_img2d(self, img, *args, **kwargs):
        """
        Args:
            img (nd array Image): Image to be scaled.

        Returns:
            nd array Image: Rescaled image.
        """
        return cv2.resize(img, tuple(list(self.size)[::-1]), interpolation=self.interpolation_input)

    def apply_to_target(self, target):
        """
        Args:
            target (nd array Image): target to be scaled.

        Returns:
            nd array Image: Rescaled target.
        """
        if self.apply_to_target:
            return cv2.resize(target, tuple(list(self.size)[::-1]), interpolation=self.interpolation_target)
        else:
            pass


class Rotate180(ProcessImage):

    def apply_to_img2d(self, img, *args, **kwargs):
        return np.rot90(np.rot90(img))

    def apply_to_target(self, target, *args, **kwargs):
        return np.rot90(np.rot90(target))


class LocalMedian(ProcessImage):

    def __init__(self, region=disk(1), background=None):
        self.region = region
        self.bg = background


    def apply_to_img2d(self, img, *args, **kwargs):
        if self.bg is not None:
            med = median(img, self.region, mask=(img != self.bg))
        else:
            med = median(img, self.region)
        res = med / med.max()      # Median is in [0, 255]
        if self.bg is not None:
            res[img == self.bg] = self.bg
        return res
        


class EqualizeHist(ProcessImage):

    def __init__(self, background=None):
        self.bg = background

    def apply_to_img2d(self, img, *args, **kwargs):
        if self.bg is not None:
            res = equalize_hist(img, mask=(img != self.bg))
            res[img == self.bg] = self.bg
        else:
            res = equalize_hist(img)
        return res


class EqualizeAdaptHist(ProcessImage):

    def __init__(self):
        pass

    def apply_to_img2d(self, img, *args, **kwargs):
        return equalize_adapthist(img)


class GaussianFilter(ProcessImage):

    def __init__(self, sigma):
        self.sigma = sigma

    def apply_to_img2d(self, img, *args, **kwargs):
        return ndimage.gaussian_filter(img, self.sigma,)


class TopHat(ProcessImage):

    def __init__(self, size_prop=.3, region_type='disk'):
        super().__init__()

    def apply_morphology(self, img, region, *args, **kwargs):
        return white_tophat(img, region)

        


