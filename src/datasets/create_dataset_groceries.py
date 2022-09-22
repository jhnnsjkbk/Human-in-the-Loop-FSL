# code based on https://github.com/cbfinn/maml

"""
Script for downloading and preprocessing the images of the Freiburg Groceries Dataset which can be downloaded here:
 http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/


Then run
    python src/datasets/create_dataset_groceries.py
"""

import csv
import glob
import os
import sys
import random
from PIL import Image

data_dir = 'datasets/groceries/'

# Download and unzip images if not done by before
if not os.path.isdir(data_dir + 'images'):
    print('Downloading images')
    # Download images
    os.system('wget -P datasets http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz')
    # unzip
    os.makedirs(data_dir, exist_ok=True)
    os.system(f'tar -xf datasets/freiburg_groceries_dataset.tar.gz -C {data_dir}')
    # remove tar file
    os.system('rm datasets/freiburg_groceries_dataset.tar.gz')

all_classes = glob.glob(data_dir + 'images/' + '*')

# Resize images to 84x84
for c, class_path in enumerate(all_classes):
    # Write progress to stdout
    sys.stdout.write(f'\r>> Resizing class {c + 1}/{len(all_classes)}')
    sys.stdout.flush()
    all_images = glob.glob(class_path + '/*')
    for i, image_file in enumerate(all_images):
        im = Image.open(image_file)
        im = im.resize((84, 84))
        im.save(image_file)

# create random split
random.seed(1234)
random.shuffle(all_classes)

for split in ['train', 'val', 'test']:
    os.makedirs(data_dir + split, exist_ok=True)

# Move classes in split directory
split_size = {'train': 10, 'val': 5, 'test': 10}
for i, class_path in enumerate(all_classes):
    # get traget folder
    if i < split_size['train']:
        new_dir = data_dir + 'train/' + os.path.basename(class_path)
    elif i < split_size['train'] + split_size['val']:
        new_dir = data_dir + 'val/' + os.path.basename(class_path)
    else:
        new_dir = data_dir + 'test/' + os.path.basename(class_path)

    os.system(f'mv {class_path} {new_dir}')
    os.system(f'rm {data_dir}images')
