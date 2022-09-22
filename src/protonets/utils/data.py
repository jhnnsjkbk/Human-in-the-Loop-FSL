import src.protonets.data


def load(cfg, splits):
    if cfg.DATASET.NAME in ['cifar100', 'miniImagenet', 'groceries', 'trafficsign', 'food101']:
        ds = src.protonets.data.dataset.load(cfg, splits)
    elif cfg.DATASET.NAME == 'omniglot':
        ds = src.protonets.data.omniglot.load(cfg, splits)
    else:
        raise ValueError(f'Unknown dataset: {cfg.DATASET.NAME:s}')

    return ds
