import os
import json
import math
from tqdm import tqdm
import argparse

import torch
import torchnet as tnt

from src.config.defaults import cfg

from src.protonets.utils import filter_opt, merge_dict
import src.protonets.utils.data as data_utils
import src.protonets.utils.model as model_utils


def get_parser():
    parser = argparse.ArgumentParser(description='Prototypical Networks Evaluation')
    parser.add_argument(
        "config_file",
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
    # set simulation specific config values
    cfg.merge_from_list([
        'DATASET.TEST_SHOT', 1,
        'DATASET.TEST_EPISODES', 1000,
    ])
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    # load model
    model = torch.load(os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.PATH))
    model.eval()

    # Some error handling
    assert cfg.MODEL == model.cfg.MODEL, f"Model was trained with different parameters\n" \
                                         f"Config: {cfg.MODEL}\nModel config: {model.cfg.MODEL}"
    # update model config with testing parameters
    model.cfg = cfg

    # Init seed for reproducibility
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
        model.cuda()

    val_log_fields = [field for field in cfg.LOG_FIELDS if field != 'lr']
    meters = {field: tnt.meter.AverageValueMeter() for field in val_log_fields}

    if not cfg.DATASET.EVAL_BASE:
        data = data_utils.load(cfg, ['test'])
        model_utils.evaluate(model, data['test'], meters, desc="test")
    else:
        data = data_utils.load(cfg, ['train'])
        model_utils.evaluate(model, data['train'], meters, desc="eval")

    for field, meter in meters.items():
        mean, std = meter.value()
        print(f'test {field:s}: {mean:0.6f} +/- {1.96 * std / math.sqrt(cfg.DATASET.TEST_EPISODES):0.6f}')


if __name__ == '__main__':
    main()
