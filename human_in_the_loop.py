import os
import math
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torchnet as tnt

from src.config.defaults import cfg

import src.protonets.utils.data as data_utils


def get_parser():
    parser = argparse.ArgumentParser(description='Human in the Loop Simulation')
    parser.add_argument(
        "config_file",
        default="configs/train_miniImagenet.yaml",
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


def plot_simulation(results, acquisition_functions, metric='Accuracy', output_path=None, title=True, plot_conf=True):
    """
    :param results:
    :param acquisition_functions:
    :param output_path:
    :param metric: str, 'accuracy', 'F1', 'ROC_AUC'
    """
    steps = np.array(range(len(results)))
    budget = steps / steps.max() * 100

    for a_func in acquisition_functions:

        # plot history of metric over all steps with 95% confidence interval
        mean = results[f'{a_func}_{metric}_mean']
        plt.plot(budget, mean, label=a_func)

        if plot_conf:
            conf = results[f'{a_func}_{metric}_95_conf']
            plt.fill_between(budget, mean - conf, mean + conf, alpha=0.2)

    plt.ylabel(metric)
    plt.xlabel('Budget [%]')
    plt.legend()
    plt.xlim(0, budget[-1])
    plt.ylim(0, 1)
    if title:
        plt.title(f'{cfg.DATASET.TEST_WAY}-Way, {cfg.DATASET.TEST_SHOT}-Shot, {cfg.DATASET.TEST_QUERY}-Query, '
                  f'{cfg.DATASET.TEST_EPISODES}-Runs')

    if output_path is None:
        plt.show()
    else:
        # plt.savefig(f'{output_path}_{metric}.png')
        plt.savefig(f'{output_path}_{metric}.pdf')
    # clear figure
    plt.clf()


def main():
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    # set simulation specific config values
    cfg.merge_from_list([
        'DATASET.TEST_SHOT', 1,
        'DATASET.TEST_EPISODES', 100,
        'DATASET.SIMULATION_TEST', 100
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

    if not cfg.DATASET.EVAL_BASE:
        # evaluate novel classes
        data = data_utils.load(cfg, ['test'])
        data_loader = tqdm(data['test'], desc='test')
    else:
        # evaluate base classes
        data = data_utils.load(cfg, ['train'])
        data_loader = tqdm(data['train'], desc='train')

    # initialize meters
    meters = {method + field: tnt.meter.AverageValueMeter()
              for field in ['_Method_Accuracy', '_Accuracy', '_Test_Accuracy', '_ROC_AUC', '_F1']
              for method in cfg.ACQUISITION_FUNCTIONS}

    for sample in data_loader:
        # simulate human-in-the-loop experiment
        output = model.human_in_the_loop(sample, cfg.ACQUISITION_FUNCTIONS)

        for field, meter in meters.items():
            meter.add(output[field])

    results = pd.DataFrame()

    for field, meter in meters.items():
        mean, std = meter.value()
        # 95 % confidence intervals
        conf = 1.96 * std / math.sqrt(cfg.DATASET.TEST_EPISODES)

        results[field + '_mean'] = mean
        results[field + '_std'] = std
        results[field + '_95_conf'] = conf

    output_path = f'{cfg.OUTPUT_DIR}{cfg.DATASET.NAME}_{cfg.DATASET.TEST_WAY}-Way_{cfg.DATASET.TEST_SHOT}-Shot_' \
                  f'{cfg.DATASET.TEST_QUERY}-Query_{cfg.DATASET.TEST_EPISODES}-Runs'
    if cfg.DATASET.EVAL_BASE:
        output_path = output_path + '_Base-classes'

    results.to_csv(f'{output_path}.csv')
    plot_simulation(results, cfg.ACQUISITION_FUNCTIONS, 'Method_Accuracy', output_path)
    plot_simulation(results, cfg.ACQUISITION_FUNCTIONS, 'Accuracy', output_path)
    plot_simulation(results, cfg.ACQUISITION_FUNCTIONS, 'Test_Accuracy', output_path)
    plot_simulation(results, cfg.ACQUISITION_FUNCTIONS, 'ROC_AUC', output_path)
    plot_simulation(results, cfg.ACQUISITION_FUNCTIONS, 'F1', output_path)


if __name__ == '__main__':
    main()
