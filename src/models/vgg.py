"""Some code taken from torch source code <https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py>"""

import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
from torchvision.models.utils import load_state_dict_from_url

from src.models.custom_layers import MaskedConv2d

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG11(vgg.VGG):

    def __init__(
        self,
        in_channels=2,
        n_classes=1,
        batch_norm=False,
        use_mask=True,
        background=-1,
        pretrained=False,
        progress=True,
        do_activation=False,
        *args,
        **kwargs
    ):
        if pretrained:
            kwargs['init_weights'] = False
        super().__init__(
            features=vgg.make_layers(cfgs['vgg11'], batch_norm=batch_norm),
        )
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['vgg11'],
                                                progress=progress)
            self.load_state_dict(state_dict)

        self.background = background

        if in_channels != 3:
            self.features[0] = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        if use_mask:
            self.features[0] = MaskedConv2d(conv2d=self.features[0], background=self.background)

        if n_classes != 1000:
            self.classifier[-1] = nn.Linear(in_features=self.classifier[-1].in_features, out_features=n_classes, bias=True)

        self.do_activation = do_activation
        self.final_activation = torch.sigmoid if n_classes == 1 else lambda x: torch.softmax(x, dim=-1)

    def forward(self, x, do_activation=None):
        if do_activation is None:
            act = self.do_activation
        else:
            act = do_activation
        x = super().forward(x)
        if act:
            return self.final_activation(x)
        return x


    def to(self, device):
        super().to(device)
        # for child in self.children():
        #     child.to(device)
        self.features[0].to(device)
