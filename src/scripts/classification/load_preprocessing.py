from os.path import join

import torchvision.transforms as transforms

import src.utils as u
import src.data_processing.processing as proc
import src.data_processing.preprocessing as prep
import src.data_processing.torch_preprocessing as tprep
import config as cfg



def load_preprocessing(save=True):

    # cfg.preprocessing = proc.ComposeProcessColumn([
    #     prep.Resize(224, do_target=False),
    #     # prep.MinMaxNorm(background=-1),
    # #     prep.LocalMedian(background=-1),
    # #     prep.EqualizeHist(background=-1),
    # ])
    cfg.preprocessing = proc.Processor()

    if save and not cfg.DEBUG:
        u.save_pickle(
            cfg.preprocessing, join(cfg.tensorboard_path, 'preprocessing.pkl'))


def load_transform_image(save=True):
    cfg.transform_image_train = transforms.Compose([
        # tprep.ToUint8(),
        # transforms.ToPILImage(),
        # transforms.RandomRotation(degrees=4) if cfg.args['DATA_AUGMENTATION'] else lambda x: x,
        transforms.Resize(224),
        transforms.ToTensor(),
        tprep.ToDevice(cfg.device),
        tprep.MaskedMinMaxNorm(background_prev=0, background_next=cfg.background),
        transforms.Normalize(
            mean=cfg.mean_norm,
            std=cfg.std_norm,
        ) if cfg.args['NORMALIZE'] else tprep.Identity(),
    ])

    cfg.transform_image_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        tprep.MaskedMinMaxNorm(background_prev=0, background_next=cfg.background),
        transforms.Normalize(
            mean=cfg.mean_norm,
            std=cfg.std_norm,
        ) if cfg.args['NORMALIZE'] else tprep.Identity(),

    ])

    if save and not cfg.DEBUG:
        u.save_pickle(
            cfg.transform_image_test,
            join(cfg.tensorboard_path, 'transform_image_test.pkl'))
        u.save_pickle(
            cfg.transform_image_train,
            join(cfg.tensorboard_path, 'transform_image_train.pkl'))


def main():
    load_preprocessing()
    load_transform_image()
