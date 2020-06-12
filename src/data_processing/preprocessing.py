import collections

import cv2
import numpy as np
from skimage.morphology import white_tophat
from skimage.exposure import equalize_adapthist, equalize_hist
from scipy import ndimage

from src.data_processing.processing import ProcessImage, GlobalMorphProcess

Iterable = collections.abc.Iterable


class ToUint8(ProcessImage):

    def __init__(self, channels=None, background=None):
        super().__init__(channels)
        self.bg = background

    def apply_to_img2d(self, img_orig,):
        if self.bg is None:
            img = (img_orig - img_orig.min()) / (img_orig.max() - img_orig.min())
            return (img * 255).astype(np.uint8)
        mask = img_orig == self.bg
        curmin = np.where(mask, img_orig.max(), img_orig).min()
        res = np.where(mask, curmin, img_orig)
        res = (res - curmin) / (res.max() - curmin)
        return np.where(mask, 0, res*254+1).astype(np.uint8)

    def apply_to_img2d_not_in_channels(self, mask,):
        if self.bg is None:
            return mask.astype(np.uint8)
        return np.where(mask == self.bg, 0, mask).astype(np.uint8)

    def apply_to_target(self, target, *args, **kwargs):
        if target is not None:
            return target.astype(np.uint8)
        else:
            return target


class NormalizeMask(ProcessImage):
    """
    Normalizing preprocesser. Normalizes depending on the intensity value of mask pixels.
    """
    def __init__(self, channels=None):
        super().__init__(channels)
        self.mean = 0
        self.std = 1

    def apply_to_img2d(self, img, *args, **kwargs):
        return (img - self.mean) / self.std

    def process_train(self, df, *args, **kwargs):
        all_means = df.apply(
            lambda x: x.pixel_array[(x.target == 1) | (x.target == 2)].mean(),
            axis=1
        )
        self.mean = all_means.mean()
        self.std = all_means.std()

        return self.apply_to_df(df)


class MinMaxNorm(ProcessImage):

    def __init__(self, background=None, channels=None):
        super().__init__(channels)
        self.bg = background

    def apply_to_img2d(self, img, *args, **kwargs):
        if self.bg is None:
            return (img - img.min()) / (img.max() - img.min())
        res = img + 0
        res[img!=self.bg] = (img[img!=self.bg] - img[img!=self.bg].min()) / (img[img!=self.bg].max() - img[img!=self.bg].min())
        return res


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
        do_target=True,
        channels=None,
        # background=None,
    ):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        super().__init__(channels)
        if type(size) in [int, np.int64]:
            size = (size, size)
        self.size = size
        self.interpolation_input = interpolation_input
        self.interpolation_target = interpolation_target
        self.do_target = do_target
        # self.bg = background

    def apply_to_img2d(self, img, interpolation, *args, **kwargs):
        """
        Args:
            img (nd array Image): Image to be scaled.

        Returns:
            nd array Image: Rescaled image.
        """
        # return cv2.resize(img, tuple(list(self.size)[::-1]), interpolation=self.interpolation_input)
        # return resize_bg(img, tuple(list(self.size)[::-1]), background=self.bg, interpolation=self.interpolation_input)
        return cv2.resize(img, tuple(list(self.size)[::-1]), interpolation=interpolation)

    def apply_to_target(self, target):
        """
        Args:
            target (nd array Image): target to be scaled.

        Returns:
            nd array Image: Rescaled target.
        """
        if self.do_target:
            return cv2.resize(target, tuple(list(self.size)[::-1]), interpolation=self.interpolation_target)
        else:
            pass

    def apply_to_row(self, row_original):
        row = row_original.copy()
        if type(self.interpolation_input) == int:
            interpolation_input = [[self.interpolation_input] for _ in range(row.pixel_array.shape[-1])]
        else:
            interpolation_input = self.interpolation_input
        row.pixel_array = self.apply_to_img(row.pixel_array, args_per_chan=interpolation_input)
        if 'target' in row.keys():
            row.target = self.apply_to_target(row.target)
        return row


class Rotate180(ProcessImage):

    def apply_to_img2d(self, img, *args, **kwargs):
        return np.rot90(np.rot90(img))

    def apply_to_target(self, target, *args, **kwargs):
        return np.rot90(np.rot90(target))


class EqualizeHist(ProcessImage):

    def __init__(self, background=None, channels=None):
        super().__init__(channels)
        self.bg = background

    def apply_to_img2d(self, img, *args, **kwargs):
        if self.bg is not None:
            res = equalize_hist(img, mask=(img != self.bg))
            res[img == self.bg] = self.bg
        else:
            res = equalize_hist(img)
        return res


class EqualizeAdaptHist(ProcessImage):

    def apply_to_img2d(self, img, *args, **kwargs):
        return equalize_adapthist(img)


class GaussianFilter(ProcessImage):

    def __init__(self, sigma, channels=None):
        super().__init__(channels)
        self.sigma = sigma

    def apply_to_img2d(self, img, *args, **kwargs):
        return ndimage.gaussian_filter(img, self.sigma,)


class TopHat(GlobalMorphProcess):

    def __init__(self, size_prop=.3, region_type='disk'):
        super().__init__()

    def apply_morphology(self, img, region, *args, **kwargs):
        return white_tophat(img, region)


class RemoveGaussianBias(ProcessImage):

    def __init__(self, coef_smoothing=.05, non_zero_divide=1e-6, channels=None):
        super().__init__(channels)
        self.coef_smoothing = coef_smoothing
        self.non_zero_divide = non_zero_divide

    def apply_to_img2d(self, img, *args, **kwargs):

        sigma = (img.shape[0] * self.coef_smoothing, img.shape[1] * self.coef_smoothing)
        cor = ndimage.gaussian_filter(img, sigma=sigma)
        return img /(cor + self.non_zero_divide)
