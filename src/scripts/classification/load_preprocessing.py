from os.path import join

import torchvision.transforms as transforms

import src.utils as u
import src.data_processing.processing as proc
import src.data_processing.preprocessing as prep
import src.data_processing.torch_preprocessing as tprep
import config as cfg



def load_preprocessing():

    # cfg.preprocessing = proc.ComposeProcessColumn([
    #     prep.Resize(224, apply_to_target=False),
    #     prep.MinMaxNorm(background=-1),
    #     prep.LocalMedian(background=-1),
    #     prep.EqualizeHist(background=-1),
    # ])
    cfg.preprocessing = proc.Processor()

    u.save_pickle(cfg.preprocessing, join(cfg.tensorboard_path, 'preprocessing.pkl'))

def load_transform_image():
    cfg.transform_image_train = transforms.Compose([
        # tprep.ToUint8(),
        # transforms.ToPILImage(),
        # transforms.RandomRotation(degrees=4) if cfg.args['DATA_AUGMENTATION'] else lambda x: x,
        transforms.ToTensor(),
        tprep.ToDevice(cfg.device),
        transforms.Normalize(
            mean=cfg.mean_norm, 
            std=cfg.std_norm,
        ) if cfg.args['NORMALIZE'] else lambda x: x,
    ])

    cfg.transform_image_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.mean_norm, 
            std=cfg.std_norm,
        ) if cfg.args['NORMALIZE'] else lambda x: x,

    ])

def main():
    load_preprocessing()
    load_transform_image()

