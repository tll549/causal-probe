from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, 
    format='%(asctime)s %(levelname)s %(name)s : %(message)s',
    datefmt='%Y%m%d %H%M%S',
    handlers=[logging.StreamHandler(), 
        logging.FileHandler('logs/all_log.log')])

import argparse

PATH_TO_VEC = 'examples/glove/glove.840B.300d.txt'

import causal_probe

from causal_probe.utils import dotdict

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import BertTokenizer, BertModel

def get_args():
    parser = argparse.ArgumentParser(description='')

    # global path parameters
    parser.add_argument('-data_path', type=str, 
        default='data/causal_probing/SemEval_2010_8/processed/SemEval_processed.txt',
                        help='path to datasets')
    parser.add_argument('-output_path', type=str, default='',
                        help='path to output results')

    # model settings
    parser.add_argument('-model', type=str, default='bert', 
        choices=['bert', 'glove', 'conceptnet', 'gpt2', 'ALL'],
        help='choose model(bert, spanbert, roberta, gpt2)')
    parser.add_argument('-model_type', type=str, default='base',
                        choices=['base', 'large', 'small', 'medium', 'large'],
                        help='bert/gpt2 model size')
    parser.add_argument('-method', type=str, default='avg',
                        choices=['avg', 'max', 'attn', 'diff', 'diff_sum', 'coherent'],
                        help='span average/span difference')
    parser.add_argument('-cased', action='store_true', help='set cased to be true')
    parser.add_argument('-fine_tune', action='store_true', help='fine tune')

    # Model configuration
    parser.add_argument('-batch_size', type=int, default=128,
                        help='batch size (default: 128)')

    # classifier parameters
    parser.add_argument('-use_pytorch', action='store_true',
                        help='use pytorch MLP classifier or sklearn logistic regression')
    # parser.add_argument('-nhid', type=int, default=0,
    #                     help='number of hidden units (0: Logistic Regression, >0: MLP); Default nonlinearity: Tanh')
    parser.add_argument('-optim', type=str, default='rmsprop',
                        help='optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)')

    # # log output
    parser.add_argument('-log_to_file', action='store_true',
                        help='log to file')


    # probing task
    parser.add_argument('-trial', action='store_true',
        help='if trial run') # '-trial' means true, '' means false
    parser.add_argument('-probe', type=str, default='simple',
        choices=['simple', 'mask', 'choice', 'feature'],
        help='types of probing task, (simpel causal, predict masked, choose between two choises)')
    parser.add_argument('-dataset', type=str, default='semeval',
        choices=['semeval', 'because', 'roc', 'biocausal'],
        help='')
    parser.add_argument('-label_data', type=str, default='oanc',
        choices=['semeval', 'oanc'],
        help='the dataset used as ground truth corpus to calc conditional probabilities')
    parser.add_argument('-reset_data', type=int, default=1,
        help='bool')
    parser.add_argument('-swap_cause_effect', action='store_true',
        help='swap cause and effect, use for simple probe')
    parser.add_argument('-subset_data', type=str, default='all',
        choices=['all', 'explicit', 'implicit', # for feature probe
            'downsampling', # for simple probe
            'explicit_down'], # for balancing explicit and implicit
        help='subset data to use only explicit or implicit')
    parser.add_argument('-reencode_data', type=int, default=1,
        help='bool')
    parser.add_argument('-seed', type=int, default=555)
    parser.add_argument('-mask', type=str, default='cause',
        choices=['cause', 'effect'],
        help='')

    parser.add_argument('-cv', type=int, default=5,
        help='number of folds to cross validation')
    parser.add_argument('-num_classes', type=int, default=5,
        help='number of classes to divide float target variable to')
    parser.add_argument('-num_classes_by', type=str, default='quantile',
        choices=['linear', 'quantile'],
        help='how to divide numerical target variables to classes (default: quantile), linear will cause imbalance data')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.debug(args)

    if args.probe == 'simple':
        params = {
            'use_pytorch': args.use_pytorch,
            'trial': args.trial,
            'probing_task': args.probe,
            'dataset': args.dataset,
            'reset_data': args.reset_data,
            'swap_cause_effect': args.swap_cause_effect,
            'reencode_data': args.reencode_data,
            'subset_data': args.subset_data,
            
            'seed': args.seed,
            'pretrained': {
                'model': args.model,
                'model_type': args.model_type,
                'cased': args.cased
            },
            'cv': args.cv,
        }
        ce = causal_probe.engine(params)
        result = ce.eval()

    elif args.probe == 'mask':
        params = {
            'trial': args.trial,
            'probing_task': args.probe,
            'dataset': args.dataset,
            'reset_data': args.reset_data,
            'seed': args.seed,
            'pretrained': {
                'model': args.model,
                'model_type': args.model_type,
                'cased': args.cased
            },
            'mask': args.mask
        }
        ce = causal_probe.engine(params)
        result = ce.eval()
        # print(result)

    elif args.probe == 'feature':
        params = {
            'use_pytorch': args.use_pytorch,
            'trial': args.trial,
            'probing_task': args.probe,
            'dataset': args.dataset,
            'reset_data': args.reset_data,
            'reencode_data': args.reencode_data,
            'label_data': args.label_data,
            'subset_data': args.subset_data,

            'seed': args.seed,
            'pretrained': {
                'model': args.model,
                'model_type': args.model_type,
                'cased': args.cased,
            },
            'cv': args.cv,
            'num_classes': args.num_classes,
            'num_classes_by': args.num_classes_by
        }
        ce = causal_probe.engine(params)
        result = ce.eval()