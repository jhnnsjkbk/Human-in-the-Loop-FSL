import os
import json
import subprocess
import argparse

from src.config.defaults import cfg

from src.protonets.utils import format_opts, merge_dict
from src.protonets.utils.log import load_trace


def get_parser():
    parser = argparse.ArgumentParser(description='Prototypical Networks Training on trainval')
    parser.add_argument(
        "config_file",
        default="configs/train_cifar100.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def main():
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # get target training loss to exceed
    trace_file = os.path.join(cfg.OUTPUT_DIR, 'trace.txt')
    trace_vals = load_trace(trace_file)
    best_epoch = trace_vals['val']['loss'].argmin()

    # load opts
    model_config_file = os.path.join(cfg.OUTPUT_DIR, 'opt.json')
    with open(model_config_file, 'r') as f:
        model_config = json.load(f)

    # override previous training ops
    model_config = merge_dict(model_config, {
        'OUTPUT_DIR': os.path.join(model_config['OUTPUT_DIR'], 'trainval'),
        'DATASET.TRAINVAL': True,
        'TRAIN.EPOCHS': best_epoch + model_config['TRAIN.PATIENCE'],
    })

    subprocess.call(['python', os.path.join(os.getcwd(), 'train.py')] + format_opts(model_config))


if __name__ == '__main__':
    main()
