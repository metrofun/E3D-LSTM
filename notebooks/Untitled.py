# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sys
import torch
import h5py
import matplotlib.pyplot as plt
from src.dataset import SequantialDataset

# %matplotlib inline

# %load_ext autoreload
# %autoreload 2

sys.path.append('..')

f = h5py.File('../data/TaxiBJ/train/BJ13_M32x32_T30_InOut.h5', 'r')
dataset = SequantialDataset(f.get('data'), 12, 3)
dataset[0][0].shape
def draw(imgs):
    size = imgs.shape[0]
    fig, axs = plt.subplots(2, size, figsize=(12, 12))
    for img, ax1, ax2 in zip(imgs, axs[0], axs[1]):
        # reorder dimensions (2, 32, 32) -> (32, 32, 2)
        img = img.permute(1,2,0)
        ax1.imshow(img[:,:,0])
        ax2.imshow(img[:,:,1])
        for ax in (ax1, ax2):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()


draw(dataset[50][0])

 
