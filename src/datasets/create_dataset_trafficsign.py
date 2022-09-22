# code based on https://github.com/cbfinn/maml

"""
Script for downloading and preprocessing the images of the GTSRB - German Traffic Sign Recognition Benchmark:
 https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign


Then run
    python src/datasets/create_dataset_trafficsign.py
"""

import csv
import glob
import os
import sys
import random
import pandas as pd
from PIL import Image

data_dir = 'datasets/trafficsign/'

# manual split to avoid classes with similar shapes being in train and test split
class_split = {'train': {0, 1, 2, 3, 4, 5, 7, 8, 11, 15, 16, 18, 19, 20, 21, 23, 24, 25, 27, 28, 29, 30, 31},
               'val': {6, 9, 14, 22, 26, 32, 33, 34, 35, 36},
               'test': {10, 12, 13, 17, 37, 38, 39, 40, 41, 42}}

if not os.path.isdir(data_dir + 'archive'):
    assert os.path.isfile(data_dir + 'archive.zip'), f'First download the dataset https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign and place archive.zip in {data_dir}'
    print('Unzipping archive.zip')
    os.system(f'unzip {data_dir}archive.zip {data_dir}archive')

# Resize images to 84x84
print('Resizing traffic sign images')
for root, dirs, files in os.walk(data_dir):
    for image_file in files:
        if image_file.endswith('.png'):
            im = Image.open(os.path.join(root, image_file))
            im = im.resize((84, 84))
            im.save(os.path.join(root, image_file))

# Create dirs
for split in ['train', 'val', 'test']:
    os.makedirs(data_dir + split, exist_ok=True)

# Move classes in split directory
for split_name, split_classes in class_split.items():
    for class_name in split_classes:
        old_dir = os.path.join(data_dir, 'archive', 'Train', str(class_name))
        new_dir = os.path.join(data_dir, split_name, str(class_name))
        os.system(f'mv {old_dir} {new_dir}')

# Delete old folders
os.system(f'rm -r {data_dir}archive')
if os.path.isfile(data_dir + 'archive.zip'):
    os.system(f'rm {data_dir}archive.zip')


