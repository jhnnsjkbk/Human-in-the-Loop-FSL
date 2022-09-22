import os
import sys
import glob

from functools import partial

import numpy as np
from PIL import Image

import torch

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

import src.protonets
from src.protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler

CIFAR100_CACHE = {}

def load_image_path(key, out_field, d):
    d[out_field] = np.array(Image.open(d[key]), np.float32, copy=False)
    return d


def subtract_means(key, mean, std, d):
    d[key] = (d[key] - np.array(mean, dtype=np.float32)) / (np.array(std, dtype=np.float32) * 255)
    return d


def convert_tensor(key, d):
    d[key] = torch.from_numpy(d[key]).transpose(2, 0)
    return d

def rotate_image(key, rot, d):
    d[key] = d[key].rotate(rot)
    return d

def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width))
    return d


def load_class_images(cfg, d):
    if d['class'] not in CIFAR100_CACHE:
        image_dir = os.path.join(cfg.DATASET.DATA_DIR, cfg.DATASET.NAME, d['class'])

        class_images = sorted(glob.glob(os.path.join(image_dir, '*.png')) +
                              glob.glob(os.path.join(image_dir, '*.jpg')))

        assert len(class_images) != 0, f"No images found for {d['class']}  class at {image_dir}"

        transform = [partial(convert_dict, 'file_name'),
                     partial(load_image_path, 'file_name', 'data')]
        if cfg.DATA_AUGMENTATION.NORMALIZE:
            # Normalize images with mean and std
            transform.append(partial(subtract_means, 'data', cfg.DATA_AUGMENTATION.PIXEL_MEAN,
                                     cfg.DATA_AUGMENTATION.PIXEL_STD))
        transform.append(partial(convert_tensor, 'data'))

        image_ds = TransformDataset(ListDataset(class_images), compose(transform))

        loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            CIFAR100_CACHE[d['class']] = sample['data']
            break # only need one sample because batch size equal to dataset length

    return {'class': d['class'], 'data': CIFAR100_CACHE[d['class']]}


def extract_episode(n_support, n_query, n_test, d):
    # data: N x C x H x W
    n_examples = d['data'].size(0)

    if n_query == -1:
        n_query = n_examples - n_support - n_test

    example_inds = torch.randperm(n_examples)[:(n_support+n_query+n_test)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:n_support+n_query]
    test_inds = example_inds[n_support + n_query:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]
    xt = d['data'][test_inds]

    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq,
        'xt': xt
    }


def load(cfg, splits):
    ret = {}
    for split in splits:
        if (split in ['val', 'test'] or cfg.DATASET.EVAL_BASE) and cfg.DATASET.TEST_WAY != 0:
            n_way = cfg.DATASET.TEST_WAY
        else:
            n_way = cfg.DATASET.WAY

        if (split in ['val', 'test'] or cfg.DATASET.EVAL_BASE) and cfg.DATASET.TEST_SHOT != 0:
            n_support = cfg.DATASET.TEST_SHOT
        else:
            n_support = cfg.DATASET.SHOT

        if (split in ['val', 'test'] or cfg.DATASET.EVAL_BASE) and cfg.DATASET.TEST_QUERY != 0:
            n_query = cfg.DATASET.TEST_QUERY
        else:
            n_query = cfg.DATASET.QUERY

        if (split in ['val', 'test'] or cfg.DATASET.EVAL_BASE) and cfg.DATASET.TEST_EPISODES != 0:
            n_episodes = cfg.DATASET.TEST_EPISODES
        else:
            n_episodes = cfg.DATASET.TRAIN_EPISODES

        if split in ['train', 'trainval'] and not cfg.DATASET.EVAL_BASE:
            print(f'Training {n_way:d}-way, {n_support:d}-shot '
                  f'with {n_query:d} query examples/class over {n_episodes:d} episodes')
        else:
            print(f'Evaluating {n_way:d}-way, {n_support:d}-shot '
                  f'with {n_query:d} query examples/class over {n_episodes:d} episodes')

        n_test = cfg.DATASET.SIMULATION_TEST

        transforms = [partial(convert_dict, 'class'),
                      partial(load_class_images, cfg),
                      partial(extract_episode, n_support, n_query, n_test)]
        if torch.cuda.is_available():
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        class_names = glob.glob(os.path.join(cfg.DATASET.DATA_DIR, cfg.DATASET.NAME, split, '*'))
        if split == 'trainval':
            class_names = glob.glob(os.path.join(cfg.DATASET.DATA_DIR, cfg.DATASET.NAME, 'train', '*'))
            class_names.extend(glob.glob(os.path.join(cfg.DATASET.DATA_DIR, cfg.DATASET.NAME, 'val', '*')))
        class_names = [c.split(f'{cfg.DATASET.NAME}/')[-1] for c in class_names]

        ds = TransformDataset(ListDataset(class_names), transforms)

        if cfg.DATASET.SEQUENTIAL:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret
