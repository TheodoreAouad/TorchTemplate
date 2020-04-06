import os

import matplotlib.pyplot as plt
import numpy as np
import pathlib




def show_results(targets, imgs, outputs=None, **kwargs):    

    fig, axs = plt.subplots(imgs.shape[0], 2 + len(outputs), squeeze=False, **kwargs)

    for i in range(imgs.shape[0]):
        axs[i, 0].imshow(imgs.cpu()[i][imgs.shape[1]//2], cmap='gray')
        axs[i, 0].set_title('Input image')
        # axs[i, 1].imshow(apply_mask(imgs.cpu()[i][imgs.shape[1]//2], targets.cpu()[i]))
        plot_img_mask_on_ax(axs[i, 1], imgs.cpu()[i][imgs.shape[1]//2], targets.cpu()[i])
        axs[i, 1].set_title('True mask')
        for idx_output, output in enumerate(outputs):
            # axs[i, idx_output+2].imshow(apply_mask(imgs.cpu()[i][imgs.shape[1]//2], output.detach().cpu().argmax(1)[i]))
            plot_img_mask_on_ax(axs[i, idx_output+2], imgs.cpu()[i][imgs.shape[1]//2], output.detach().cpu().argmax(1)[i])
            axs[i, idx_output+2].set_title('Predicted mask')
    
    return fig


def plot_img_mask_on_ax(ax, img, mask, alpha=.7):

    masked = np.ma.masked_where(mask == 0, mask)
    ax.imshow(img, cmap='gray')
    ax.imshow(masked, cmap='jet', alpha=alpha)


def save_figs(figs, path, filename=''):

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    all_imgs = [int(img.split('--')[-1][:-4]) for img in os.listdir(path) if ('.png' in img) and (filename in img)]
    first = max(all_imgs) + 1 if len(all_imgs) != 0 else 0
    for i, fig in enumerate(figs):
        fig.savefig(os.path.join(path, '{}--{}.png'.format(filename.split('--')[0], i + first)))
        print('saved in ', os.path.join(path, '{}--{}.png'.format(filename.split('--')[0], i + first)))

