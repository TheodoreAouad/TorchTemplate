import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np



class CustomDataset(Dataset):

    def __init__(self,
        data,
        transform_image=transforms.ToTensor(),
        tensor_type=torch.float32,
        target_type=torch.LongTensor,
        *args,
        **kwargs
    ):
        """
        Custom Dataset for images. Image Array in pixel_array column, target in taret column.
        """

        self.data = data
        self.transform_image = transform_image
        self.tensor_type = tensor_type
        self.target_type = target_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.pixel_array.iloc[idx]
        target = self.data.target.iloc[idx].astype(np.uint8)

        # Fix negative stride (e.g. matrice[..., ::-1])
        image = image - np.zeros_like(image)
        target = target - np.zeros_like(target)

        if self.transform_image is not None:
            image = self.transform_image(image)

        image, target = image.type(self.tensor_type), torch.tensor([target]).type(self.target_type)

        return image, target


MNIST = torchvision.datasets.MNIST