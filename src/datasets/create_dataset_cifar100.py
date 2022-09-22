# code based on https://github.com/cbfinn/maml and https://github.com/ElementAI/TADAM

"""
Script for downloading Cifar100 dataset and creating the few-shot dataset in dataset/cifar100/ proposed by
Oreshkin et al. (2018). TADAM: Task dependent adaptive metric for improved few-shot learning.

Run
    python src/datasets/create_dataset_cifar100.py
"""

import numpy as np
import pickle
import os
from PIL import Image

# Dataset split
# 00 'aquatic_mammals'
# 01 'fish'
# 02 'flowers'
# 03 'food_containers'
# 04 'fruit_and_vegetables'
# 05 'household_electrical_devices'
# 06 'household_furniture'
# 07 'insects'
# 08 'large_carnivores'
# 09 'large_man-made_outdoor_things'
# 10 'large_natural_outdoor_scenes'
# 11 'large_omnivores_and_herbivores'
# 12 'medium_mammals'
# 13 'non-insect_invertebrates'
# 14 'people'
# 15 'reptiles'
# 16 'small_mammals'
# 17 'trees'
# 18 'vehicles_1'
# 19 'vehicles_2'

# class split from Oreshkin et al. (2018) https://github.com/ElementAI/TADAM
class_split = {'train': {1, 2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19}, 'val': {8, 11, 13, 16}, 'test': {0, 7, 12, 14}}
# Make train, validation and test splits deterministic from one run to another
np.random.seed(2017 + 5 + 17)

input_dir = 'datasets/cifar-100-python'
output_dir = 'datasets/cifar100'

# automatic download and extraction of Cifar100 dataset
if not os.path.isdir(input_dir):
    # download cifar100
    os.system('wget -P datasets https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz')
    # extract tar
    os.system('tar -xf datasets/cifar-100-python.tar.gz -C datasets')
    # remove tar file
    os.system('rm datasets/cifar-100-python.tar.gz')

# load the full Cifar100 dataset, including train and test
with open(os.path.join(input_dir, 'train'), 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
    images = dict[b'data']
    fine_labels = dict[b'fine_labels']
    coarse_labels = dict[b'coarse_labels']

with open(os.path.join(input_dir, 'test'), 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
    images = np.concatenate((images, dict[b'data']))
    fine_labels = np.concatenate((fine_labels,dict[b'fine_labels']))
    coarse_labels = np.concatenate((coarse_labels,dict[b'coarse_labels']))

images = images.reshape((-1, 3, 32, 32))
images = images.transpose((0, 2, 3, 1))

for split_name, split_coarse_classes in class_split.items():
    split_images = []
    split_fine_labels = []
    split_coarse_labels = []

    # get all images with split_coarse_classes
    for current_coarse_label in split_coarse_classes:
        idxs = coarse_labels == current_coarse_label
        split_images.append(images[idxs])
        split_fine_labels.append(fine_labels[idxs])
        split_coarse_labels.append(coarse_labels[idxs])

    # concat all images in current split
    split_images = np.concatenate(split_images)
    split_fine_labels = np.concatenate(split_fine_labels)
    split_coarse_labels = np.concatenate(split_coarse_labels)

    current_label = ''
    label_dir = ''
    # save all images of a label in one dir
    for i, (img, fine_label, coarse_label) in enumerate(zip(split_images, split_fine_labels, split_coarse_labels)):
        if current_label != f'{coarse_label}_{fine_label}':
            current_label = f'{coarse_label}_{fine_label}'
            # create label directory
            label_dir = os.path.join(output_dir, split_name, current_label)
            os.makedirs(label_dir, exist_ok=True)
        # save image as png
        img = Image.fromarray(img)
        img.save(f'{label_dir}/{current_label}_{i}.png')

    print(f'Saved {split_name} images')

# Remove downloaded dataset after extracting few-shot-learning split
os.system('rm -r datasets/cifar-100-python')
