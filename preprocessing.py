import os
import pandas as pd
import cv2
from PIL import Image
import math
from skimage import io, color
import torch


# Get mean and standard deviation of training data for normalisation
def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, _, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    return mean.tolist(), std.tolist()


# Centre crops and resizes images and saves to new location
def crop_resize(rootdir, savedir, size):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            im = cv2.imread(os.path.join(subdir, file))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # if image wider than long centre crop by smallest side
            if im.shape[1] < im.shape[0]:
                dim1 = int(im.shape[1])
                dim0 = int(im.shape[0])
                chi = int((dim0 - dim1) / 2)
                im = im[chi:chi + dim1, 0:dim1]
            else:
                # if image longer than wide centre crop by smallest side
                dim1 = int(im.shape[0])
                dim0 = int(im.shape[1])
                chi = int((dim0 - dim1) / 2)
                im = im[0:dim1, chi:chi + dim1]

            im = Image.fromarray(im, 'RGB')
            # resizing
            im = im.resize((size, size), Image.ANTIALIAS)
            im.save(os.path.join(savedir, file))
    print("Done")


# Calculates image dimensions from raw images and saves to dataframe
def get_size_from_raw(df):
    def sizeify(filepath):
        image = Image.open(filepath)
        width, height = image.size
        return f'{width}x{height}'
    df['size'] = df['filepath'].apply(lambda x: sizeify(x))
    return df


# Use on ISIC dataframe to get dimensions as single variable
def get_size_ISIC(df):
    df['size'] = 0

    def sizeify(width, height):
        return f'{width}x{height}'

    df['size'] = df.apply(lambda x: sizeify(df.width, df.height), axis=1)
    return df
