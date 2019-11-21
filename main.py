# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import argparse

# Set PATHs
# PATH_TO_SENTEVAL = '.'
# PATH_TO_DATA = 'data'
PATH_TO_VEC = 'examples/glove/glove.840B.300d.txt'

# import SentEval
# sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# from reference.encoder import Encoder
from utils import dotdict

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import BertTokenizer, BertModel

def get_args():
    '''https://github.com/lingyugao/causal/blob/master/main.py#L101'''
    parser = argparse.ArgumentParser(description='')

    # global path parameters
    parser.add_argument('-data_path', type=str, default='data',
                        help='path to datasets')
    parser.add_argument('-output_path', type=str, default='',
                        help='path to output results')
    parser.add_argument('-seed', type=int, default=0,
                        help='Random seed (default: 0)')

    # model settings
    parser.add_argument('-model', type=str, default='bert', 
        # choices=['bert', 'spanbert', 'roberta', 'xlnet', 'gpt2', 'glove'],
        choices=['bert', 'glove'],
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
    # parser.add_argument('-nhid', type=int, default=0,
    #                     help='number of hidden units (0: Logistic Regression, >0: MLP); Default nonlinearity: Tanh')
    parser.add_argument('-optim', type=str, default='rmsprop',
                        help='optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)')

    # reset data or not
    parser.add_argument('-reset_data', action='store_true', help='reset data instead of loading')
    parser.add_argument('-rerun', action='store_true',
                        help='rerun this setting whether it was ran before')

    # specify task
    # parser.add_argument('-transfer_task', type=str, nargs='*', default='Length',
    #                     help='Length, WordContent, Depth, TopConstituents, BigramShift, '
    #                          'Tense, SubjNumber, ObjNumber, OddManOut, CoordinationInversion')

    # # log output
    # parser.add_argument('-log_to_file', action='store_true',
    #                     help='log to file')
    #
    # # output settings
    # parser.add_argument('-output_test', action='store_true',
    #                     help='use test set and output the result')
    return parser.parse_args()


# SentEval prepare and batcher
def prepare(params, samples):
    '''
    sees the whole dataset of each task and can thus construct the word vocabulary, 
    the dictionary of word vectors etc

    batcher only sees one batch at a time while the samples argument of prepare contains 
        all the sentences of a task.

    params: senteval parameters.
    samples: list of all sentences from the tranfer task.
    output: No output. Arguments stored in "params" can further be used by batcher.

    Example: in bow.py, prepare is is used to build the vocabulary of words and construct 
        the "params.word_vect* dictionary of word vectors.
        samples = self.task_data['train']['X'] + self.task_data['dev']['X'] + \
                      self.task_data['test']['X']
        modified (add, as below):
            params.word2id, params.word_vec, wvec_dim
    '''

    '''
    prepare the encoder (BERT) given pars
    will be used in batcher to encode and compute sentence embeddings
    no need samples here
    '''
    logging.info('preparing...')

    params.pretrained = dotdict(params.pretrained)
    if params.pretrained.model == 'bert':
        bert_type = f'bert-{params.pretrained.model_type}-{"cased" if params.pretrained.cased else "uncased"}'
        params.tokenizer = BertTokenizer.from_pretrained(bert_type)
        params.encoder = BertModel.from_pretrained(bert_type).to(device)
        params.encoder.eval() # ??

    elif params.pretrained.model == 'glove':
        # print(len(samples), samples[0])
        samples = [s.split() for s in samples]
        # Create dictionary
        def create_dictionary(sentences, threshold=0):
            words = {}
            for s in sentences:
                for word in s:
                    words[word] = words.get(word, 0) + 1

            if threshold > 0:
                newwords = {}
                for word in words:
                    if words[word] >= threshold:
                        newwords[word] = words[word]
                words = newwords
            words['<s>'] = 1e9 + 4
            words['</s>'] = 1e9 + 3
            words['<p>'] = 1e9 + 2

            sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
            id2word = []
            word2id = {}
            for i, (w, _) in enumerate(sorted_words):
                id2word.append(w)
                word2id[w] = i

            return id2word, word2id
        _, params.word2id = create_dictionary(samples)
        # Get word vectors from vocabulary (glove, word2vec, fasttext ..)
        def get_wordvec(path_to_vec, word2id):
            word_vec = {}

            with io.open(path_to_vec, 'r', encoding='utf-8') as f:
                # if word2vec or fasttext file : skip first line "next(f)"
                for line in f:
                    word, vec = line.split(' ', 1)
                    if word in word2id:
                        word_vec[word] = np.fromstring(vec, sep=' ')

            logging.info('Found {0} words with word vectors, out of \
                {1} words'.format(len(word_vec), len(word2id)))
            return word_vec
        params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
        params.wvec_dim = 300
        # print(params.word2id['book']) # 4158
        # print(len(params.word_vec['book'])) # 300
        # print(params.word_vec['book']) # 

    logging.info('prepared')
    return

def batcher(params, batch):
    '''
    transforms a batch of text sentences into sentence embeddings

    params: senteval parameters.
    batch: numpy array of text sentences (of size params.batch_size)
    output: numpy array of sentence embeddings (of size params.batch_size)
    Example: in bow.py, batcher is used to compute the mean of the word vectors for each 
        sentence in the batch using params.word_vec. Use your own encoder in that function 
        to encode sentences.
    '''
    # print("\nbatcher\n")

    if params.pretrained.model == 'bert':
        tokenizer = params.tokenizer
        encoder = params.encoder
        hidden_size = encoder.config.hidden_size

        embeddings = torch.zeros(len(batch), hidden_size).to(device)
        with torch.no_grad():
            for i in range(len(batch)): # for each sentence
                sentence = batch[i]
                # sentence = '[CLS] Jim Henson was a puppeteer? [SEP]'
                # print(sentence)
                tokenized_text = tokenizer.tokenize(sentence)
                # print(tokenized_text) # ['[CLS]', 'jim', 'henson', 'was', 'a', 'puppet', '##eer', '?', '[SEP]']
                tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)]).to(device)
                # print(tokens_tensor) # tensor([[  101,  3958, 27227,  2001,  1037, 13997, 11510,  1029,   102]])

                outputs = encoder(tokens_tensor)
                encoded_layers = outputs[0]
                # print(torch.mean(encoded_layers, dim=1).shape) # (batch_size, hidden_size) = (1, 768)
                embeddings[i, :] = torch.mean(encoded_layers, dim=1)
        # print(embeddings.shape) # (params.batch_size, ) = (128, 768)
        # return 
        return embeddings.cpu()

    elif params.pretrained.model == 'glove':
        embeddings = []
        for sent in batch:
            sentvec = []
            for word in sent:
                if word in params.word_vec:
                    sentvec.append(params.word_vec[word])
            if not sentvec:
                vec = np.zeros(params.wvec_dim)
                sentvec.append(vec)
            sentvec = np.mean(sentvec, 0)
            embeddings.append(sentvec)

        embeddings = np.vstack(embeddings)
        # print(embeddings.shape) # (128, 300)
        # print(embeddings)
        return embeddings


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
# logging.basicConfig(filename=log_file, filemode=file_mode_dict[args.rerun],
#                             level=logging.INFO, format='%(asctime)s %(message)s',
#                             datefmt='%m-%d %H:%M')

if __name__ == '__main__':
    args = get_args()
    logging.debug(args)

    # par settings from lingyu
    params_senteval = {'task_path': args.data_path,
                           'output_path': args.output_path,
                           'rerun': args.rerun,
                           'reset_data': args.reset_data,
                           'batch_size': args.batch_size,

                        # 'usepytorch': True, 'kfold': 5, # from example

                           'pretrained': {'model': args.model,
                                          'model_type': args.model_type,
                                          'cased': args.cased,
                                          'fine_tune': args.fine_tune,
                                          'method': args.method},
                           'classifier': {'optim': args.optim,
                                          'nhid': 0,
                                          'noreg': True}
                           }

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['Length']
    transfer_tasks = 'SimpelCausal'
    results = se.eval(transfer_tasks)
    print(results)