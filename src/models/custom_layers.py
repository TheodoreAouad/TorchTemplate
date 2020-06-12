import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Module):
    """
    This layer performs a conv2d then masks the output. All the outputs that touched the
    background are put to 0.
    """
    def __init__(
        self,
        conv2d,
        bg_in=-1,
        bg_out=0,
    ):
        super().__init__()
        self.bg_in = bg_in
        self.bg_out = bg_out

        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        self.kernel_size = conv2d.kernel_size
        self.stride = conv2d.stride
        self.padding = conv2d.padding
        self.bias = conv2d.bias is None

        self.conv2d = conv2d

    def forward(self, x):

        xout = self.conv2d(x)
        if self.bg_in is None:
            return xout

        mask_in = (x == self.bg_in).float().to(self.device).detach()
        kern = torch.ones((1, x.shape[1], *self.kernel_size)).to(self.device)
        mask_out = (F.conv2d(input=mask_in, weight=kern, stride=self.stride, padding=self.padding) > 0).to(self.device).detach()
        return (xout - self.bg_out) * (~ mask_out) + self.bg_out

    @property
    def device(self):
        for p in self.parameters():
            return p.device


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class MaskedBatchNorm2d(nn.BatchNorm2d):
    """
    Base taken from <https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py>.
    """
    def __init__(self, num_features, bg_in=None, bg_out=None, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats,
        )
        self.bg_in = bg_in
        self.bg_out = bg_out


    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.bg_in is not None:
            mask = input != self.bg_in
        else:
            mask = torch.ones_like(input)

        if self.training:

            # calculate running estimates
            sum_chan = mask[:, 0, ...].sum((1, 2)).float().detach()

            # Computing mean of foreground
            mean = (mask * input).sum([2, 3])/sum_chan[:, None]
            mean = mean.mean(0)

            # Computing var of foreground
            var = (input - mean[None, :, None, None])**2
            var = (mask * var).sum([2, 3])/sum_chan[:, None]
            var = var.mean(0)

            # TODO: use biased var
            # var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = ((input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps)))
        if self.affine:
            input = (input * self.weight[None, :, None, None] + self.bias[None, :, None, None])

        if self.bg_out is not None:
            input = (input - self.bg_out) * mask + self.bg_out      # put all elements of mask to bg_out faster than with input[mask]

        return input


    def extra_repr(self):
        return '{num_features}, bg_in={bg_in}, bg_out={bg_out}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input
