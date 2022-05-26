#%%
from __future__ import print_function, division
import os
import tensorflow as tf
from numpy.lib.function_base import gradient
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from sklearn.impute import SimpleImputer
from nilearn import image
import os
import nibabel as nib
import glob
from PIL import Image
from torch.utils.data.dataset import Dataset
import sklearn as skl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import KFold
from freesurfer_stats import CorticalParcellationStats
from catalyst import dl
from sklearn.metrics import mean_absolute_percentage_error

np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
#creating dataset
#TODO: change back below code to all HARDDRIVES
image_list = glob.glob(
    '/data/users2/vjoshi6/Data/MRIDataConverted2/*_3T_T1w_MPR1.nii/*.gz')
stat_list_r = glob.glob(
    '/data/hcp-plis/hdd*/*/T1w/*/stats/rh.aparc.stats')
stat_list_l = glob.glob(
    '/data/hcp-plis/hdd*/*/T1w/*/stats/lh.aparc.stats')
segmentation_paths = glob.glob(
    '/data/users2/kwang26/afedorov_T1_c_data/*')

segmentation_paths_final = []


for csv in segmentation_paths:
    df = pd.read_csv(csv)
    df = df.iloc[:, 1]
    print(df)
    for path in df:
        segmentation_paths_final.append(path)

print('Segmentation paths: \n{}'.format(segmentation_paths_final))
print("Length of segmentation paths: {}".format(len(segmentation_paths_final)))

#getting both hemispheres and checking that they are matched up correctly
volume_list = []
for i in range(len(stat_list_r)):
    stat_r = stat_list_r[i]
    stat_l = stat_list_l[i]

    stat_r_split = stat_r.split('/')
    stat_l_split = stat_l.split('/')

    stat_r_path = stat_r_split[4]
    stat_l_path = stat_l_split[4]

    print("Stat_r value and stat_l value being compared: {}, {}".format(stat_r_path, stat_l_path))

    if stat_r_path == stat_l_path:
        pass
    else:
        print("Stat_r value and stat_l value being compared: {}, {}".format(stat_r_path, stat_l_path))
        raise ValueError('Paths are not matching up')

    volume_list.append((stat_r, stat_l))

#this for loop just makes it so the images and the labels are alligned with
#the same indexes
volume_list_ordered = []
for i in range(len(image_list)):
    image = image_list[i].split('/')[6][0:6]
    for value in volume_list:
        single_path = value[0]
        single_path_split = single_path.split('/')
        path = single_path_split[4]

        if (path == image):
            volume_list_ordered.append(value)

image_list_segmentation = []
volume_list_segmentation = []
segmentation_list_ordered = []
for i in range(len(image_list)):
    image = image_list[i].split('/')[6][0:6]
    for segmentation_path in segmentation_paths_final:
        segmentation_path_split = segmentation_path.split('/')
        brain_scan_number = segmentation_path_split[5]
        if image == brain_scan_number:
            segmentation_list_ordered.append(segmentation_path)
            image_list_segmentation.append(image_list[i])
            volume_list_segmentation.append(volume_list_ordered[i])

d = {'images': image_list_segmentation, 'gray_matter': volume_list_segmentation, 'segmentation':segmentation_list_ordered}
df = pd.DataFrame(d)

print("Df head: \n{}".format(df.head()))

#code from original that will generate needed CSVs 
#NOTE: names should be images, segmentation_labels, and volume_labels; will have to rename these things in Kevin's code
train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])
train.to_csv('train.csv')
validate.to_csv('validate.csv')
test.to_csv('test.csv')