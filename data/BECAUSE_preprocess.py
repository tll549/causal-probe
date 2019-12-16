import re
import logging
import argparse
import random
import os
import glob
import nltk

import pandas as pd
from causal_probe import utils

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-raw_data_path', type=str, 
        default='data/causal_probing/BECAUSE',
        help='')
    # parser.add_argument('-source', type=str, 
    #     default='CongressionalHearings',
    #     help='')
    parser.add_argument('-save_data_path', type=str, 
        default='data/causal_probing/BECAUSE/processed',
        help='')
    return parser.parse_args()

class DataLoader(object):
    def __init__(self):
        self.txt, self.ann = [], []
        self.X, self.y = [], []
        self.pos_samples = []

    def read(self, path):
        all_txt = glob.glob(os.path.join(path + '/CongressionalHearings/*.txt')) + \
            glob.glob(os.path.join(path + '/MASC/*.txt'))
        all_ann = glob.glob(os.path.join(path + '/CongressionalHearings/*.ann')) + \
            glob.glob(os.path.join(path + '/MASC/*.ann'))
        file_name = [f[:-4] for f in all_txt]
        assert all([f + '.ann' in all_ann for f in file_name]), 'every txt file should correspond to one ann file'

        for one_txt in all_txt:
            with open(one_txt, 'r') as f:
                self.txt.append(f.read())
            with open(one_txt[:-4] + '.ann', 'r') as f:
                self.ann.append(f.read())
        
        logging.info(f'{len(self.txt)} files loaded') 

    def preprocess(self):
        for txt, ann in zip(self.txt, self.ann):
            ann = [a.split('\t') for a in ann.split('\n')]
            # extract tags that are causal
            causal_ann = [a for a in ann if a != [''] and a[1][:12] == 'Consequence:' and \
                ('Cause' in a[1] or 'Effect' in a[1])]
            # keep a list of tags [consequence, cause, effect] or [consequence, effect, cause], cause and effect can be skipped
            causal_tags = [[x.split(':')[1] for x in ca[1].split()] for ca in causal_ann]
            # a dict of mapping from tag lie T123 to idx like (100, 103)
            tag2idx = {a[0]:a[1] for a in ann if a != [''] and a[0][0] == 'T'}
            tag2idx = {k:[tuple([int(y) for y in x.split(',')]) \
                for x in ','.join(v.split(' ')[1:]).split(';')][0] \
                for k, v in tag2idx.items()}
            # map all tags to idx
            causal_tags_idx = [[tag2idx[tag] for tag in ct] for ct in causal_tags]

            # process paragraph (call it sentences here)
            txt = re.sub(r'\n\t\t\t\t', '     ', txt)
            txt = re.sub(r'\n\t\t\t',   '\n   ', txt)
            txt = re.sub(r'\t', ' ', txt)

            all_br = [i for i in range(len(txt)) if txt[i] == '\n']
            sentence_start_end = [(br1+1, br2) for br1, br2 in zip([0] + all_br[:-1], all_br)] # find the index of where a sentence start and end
            sentences = [[i[0], i[1], txt[i[0]:i[1]]] for i in sentence_start_end] # in the format of [start_idx, end_idx, sentence]
            sentences = [s for s in sentences if s[2] != '']

                # print(sentences)
            # return

            # process at the level of sentence, not paragraph like the original doc
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            for paragraph in sentences:
                if paragraph[2][-1] == ':': continue # ignore like "The Chairman:"
                end = paragraph[0]
                for sent in tokenizer.tokenize(paragraph[2]):
                    # if 'has eliminated its' in sent:
                    #     print(sentences, end='\n\n')
                    start = end
                    end = start + len(sent) + 1
                    y = 0
                    for causal_tuple in causal_tags_idx:
                        if all([tag_idx[0] > start and tag_idx[1] < end for tag_idx in causal_tuple]): # means all tags in that causal relations are in this sentence
                            y = 1
                            self.pos_samples.append((len(causal_tuple), 
                                txt[causal_tuple[0][0]:causal_tuple[0][1]], causal_tuple, sent))
                            break
                    self.y.append(y)
                    # remove excess whitespaces
                    sent = ' '.join(sent.split())
                    self.X.append(sent)

            # print(self.X)
            # print(self.y)
            # print(self.pos_samples)
            # print([p[1] for p in self.pos_samples])
            # break

        logging.info(f'number of y == 1: {sum(self.y)}, 0: {len(self.y)-sum(self.y)}')
        self.output = pd.DataFrame()
        self.output['X'] = self.X
        self.output['causal'] = self.y

    def split(self, dev_prop=0.2, test_prop=0.2, seed=555):
        random.seed(seed)
        idx = list(range(len(self.X)))
        random.shuffle(idx)
        idx_1 = int(len(self.X) * (1-dev_prop-test_prop))
        idx_2 = idx_1 + int(len(self.X) * dev_prop)

        self.train_idx = idx[:idx_1]
        self.dev_idx = idx[idx_1:idx_2]
        self.test_idx = idx[idx_2:]
        logging.info(f'data splitted train: {len(self.train_idx)}, dev: {len(self.dev_idx)}, test: {len(self.test_idx)}')
        assert len(self.train_idx) + len(self.dev_idx) + len(self.test_idx) == len(self.X)

    def write(self, data_path):
        with open(data_path, 'w+') as f:
            for i in self.train_idx:
                f.write(f'tr\t{self.y[i]}\t[CLS] {self.X[i]} [SEP]\n')
            for i in self.dev_idx:
                f.write(f'va\t{self.y[i]}\t[CLS] {self.X[i]} [SEP]\n')
            for i in self.test_idx:
                f.write(f'te\t{self.y[i]}\t[CLS] {self.X[i]} [SEP]\n')
        logging.info(f'data wrote')

    def save_output(self, save_path):
        utils.save_dt(self.output, save_path, index=False)


logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    args = get_args()
    # print(args)

    dl = DataLoader()
    dl.read(args.raw_data_path)
    dl.preprocess()
    dl.split()
    dl.write(args.save_data_path + '/because_all.txt')
