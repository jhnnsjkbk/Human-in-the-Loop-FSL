# code based on https://github.com/cbfinn/maml

"""
Script for preprocessing and splitting the images of the Food Images (Food-101) dataset:
 https://www.kaggle.com/kmader/food41

Place the downloaded and place "archive.zip" under datasets/food101

Then run
    python src/datasets/create_dataset_food101.py
"""

import csv
import glob
import os
import sys
import random
import pandas as pd
from PIL import Image

data_dir = 'datasets/food101/'

if not os.path.isdir(data_dir + 'archive'):
    assert os.path.isfile(data_dir + 'archive.zip'), f'First download the dataset from https://www.kaggle.com/kmader/food41 and place archive.zip in {data_dir}'
    print('Unzipping archive.zip')
    os.system(f'unzip {data_dir}archive.zip {data_dir}archive')

# Resize images to 84x84
print('Resizing food101 images')
for root, dirs, files in os.walk(data_dir):
    for image_file in files:
        if image_file.endswith('.jpg'):
            im = Image.open(os.path.join(root, image_file))
            im = im.resize((84, 84)).convert('RGB')
            im.save(os.path.join(root, image_file.replace('jpg', 'png')))

# create random split
random.seed(1234)
all_classes = glob.glob(data_dir + 'archive/images/' + '*')
random.shuffle(all_classes)

# Create dirs
for split in ['train', 'val', 'test']:
    os.makedirs(data_dir + split, exist_ok=True)

# Move classes in split directory
split_size = {'train': 65, 'val': 16, 'test': 20}  # in total 101 classes
for i, class_path in enumerate(all_classes):
    # get traget folder
    if i < split_size['train']:
        new_dir = data_dir + 'train/' + os.path.basename(class_path)
    elif i < split_size['train'] + split_size['val']:
        new_dir = data_dir + 'val/' + os.path.basename(class_path)
    else:
        new_dir = data_dir + 'test/' + os.path.basename(class_path)

    os.system(f'mv {class_path} {new_dir}')

# Delete old folders and test images (not used)
os.system(f'rm -r {data_dir}archive')
os.system(f'rm {data_dir}archive.zip')

# Delete false image
os.system(f'rm datasets/food101/train/lasagna/3787908.png')