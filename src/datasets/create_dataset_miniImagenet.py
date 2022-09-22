# code based on https://github.com/cbfinn/maml

"""
Script for converting from csv file datafiles to a directory for each image (which is how it is loaded by MAML code)
Acquire miniImagenet from Ravi & Larochelle '17, along with the train, val, and test csv files. Put the
csv files in the miniImagenet directory and put the images in the directory 'miniImagenet/images/'.
Then run
    python src/datasets/create_dataset_miniImagenet.py
"""

import csv
import glob
import os
import sys
from PIL import Image

data_dir = 'datasets/miniImagenet/'

# Unzip images.zip if not done by before
if not os.path.isdir(data_dir + 'images'):
    assert os.path.isfile(data_dir + 'images.zip'), f'No images in {data_dir}images'
    print('Unzipping images.zip')
    os.system(f'unzip {data_dir}images.zip {data_dir}')

all_images = glob.glob(data_dir + 'images/' + '*')

# Resize images
for i, image_file in enumerate(all_images):
    im = Image.open(image_file)
    im = im.resize((84, 84), resample=Image.LANCZOS)
    im.save(image_file)
    # Write progress to stdout
    sys.stdout.write(f'\r>> Resizing image {i+1}/{len(all_images)}')
    sys.stdout.flush()

# Put in correct directory
for datatype in ['train', 'val', 'test']:
    os.system('mkdir ' + data_dir + datatype)

    with open((data_dir + datatype + '.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        last_label = ''
        for i, row in enumerate(reader):
            if i == 0:  # skip the headers
                continue
            label = row[1]
            image_name = row[0]
            if label != last_label:
                cur_dir = data_dir + datatype + '/' + label + '/'
                os.system('mkdir ' + cur_dir)
                last_label = label
            os.system('mv ' + data_dir + 'images/' + image_name + ' ' + cur_dir)

    print(f'Moved {datatype} images')