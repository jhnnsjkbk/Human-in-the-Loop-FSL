import os
import json
import argparse
from functools import partial
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchnet as tnt

from src.protonets.engine import Engine

import src.protonets.utils.data as data_utils
import src.protonets.utils.model as model_utils
import src.protonets.utils.log as log_utils

from src.config.defaults import cfg


def get_parser():
    parser = argparse.ArgumentParser(description='Prototypical Networks Training')
    parser.add_argument(
        "config_file",
        help="path to config file",
        metavar="FILE",
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

    if not os.path.isdir(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # save config
    with open(os.path.join(cfg.OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(cfg, f)
        f.write('\n')

    trace_file = os.path.join(cfg.OUTPUT_DIR, 'trace.txt')

    # Init seed for reproducibility
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)

    if cfg.DATASET.TRAINVAL:
        data = data_utils.load(cfg, ['trainval'])
        train_loader = data['trainval']
        val_loader = None
    else:
        data = data_utils.load(cfg, ['train', 'val'])
        train_loader = data['train']
        val_loader = data['val']

    if cfg.MODEL.PRETRAINED != '':
        print(f'Loading model from {cfg.MODEL.PRETRAINED}')
        model = torch.load(cfg.MODEL.PRETRAINED)
        assert model.cfg.MODEL.INPUT_DIM == cfg.MODEL.INPUT_DIM, f'Pretrained model has different image size:' \
                                                                 f'model: {model.cfg.MODEL.INPUT_DIM}; ' \
                                                                 f'config: {cfg.MODEL.INPUT_DIM}'
        model.cfg = cfg
    else:
        model = model_utils.load(cfg)

    if torch.cuda.is_available():
        model.cuda()

    engine = Engine()

    meters = {'train': {field: tnt.meter.AverageValueMeter() for field in cfg.LOG_FIELDS}}

    if val_loader is not None:
        val_log_fields = [field for field in cfg.LOG_FIELDS if field != 'lr']
        meters['val'] = {field: tnt.meter.AverageValueMeter() for field in val_log_fields}

    def on_start(state):
        if os.path.isfile(trace_file):
            os.remove(trace_file)
        state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], cfg.TRAIN.DECAY_EVERY, gamma=0.5)

    engine.hooks['on_start'] = on_start

    def on_start_epoch(state):
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()

    engine.hooks['on_start_epoch'] = on_start_epoch

    def on_update(state):
        for field, meter in meters['train'].items():
            meter.add(state['output'][field])

    engine.hooks['on_update'] = on_update

    def on_end_epoch(hook_state, state):
        state['scheduler'].step()

        if val_loader is not None:
            if 'best_loss' not in hook_state:
                hook_state['best_loss'] = np.inf
            if 'wait' not in hook_state:
                hook_state['wait'] = 0

        if val_loader is not None:
            model_utils.evaluate(state['model'],
                                 val_loader,
                                 meters['val'],
                                 desc="Epoch {:d} valid".format(state['epoch']))

        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = state['epoch']
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        if val_loader is not None:
            if meter_vals['val']['loss'] < hook_state['best_loss']:
                hook_state['best_loss'] = meter_vals['val']['loss']
                print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))

                state['model'].cpu()
                torch.save(state['model'], os.path.join(cfg.OUTPUT_DIR, 'best_model.pt'))
                if torch.cuda.is_available():
                    state['model'].cuda()

                hook_state['wait'] = 0
            else:
                hook_state['wait'] += 1

                if hook_state['wait'] > cfg.TRAIN.PATIENCE:
                    print(f'==> patience {cfg.TRAIN.PATIENCE:d} exceeded')
                    state['stop'] = True
        else:
            state['model'].cpu()
            torch.save(state['model'], os.path.join(cfg.OUTPUT_DIR, 'best_model.pt'))
            if torch.cuda.is_available():
                state['model'].cuda()

    engine.hooks['on_end_epoch'] = partial(on_end_epoch, {})

    engine.train(
        model=model,
        loader=train_loader,
        optim_method=getattr(optim, cfg.TRAIN.OPTIMIZER),
        optim_config={'lr': cfg.TRAIN.LEARNING_RATE,
                      'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
        max_epoch=cfg.TRAIN.EPOCHS
    )


if __name__ == '__main__':
    main()
