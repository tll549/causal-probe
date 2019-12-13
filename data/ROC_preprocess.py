import re
import logging
import argparse
import random
import os
import glob
import nltk
from nltk.tokenize import word_tokenize
from zipfile import ZipFile
import codecs
import pandas as pd
from causal_probe import utils

class DataLoader(object):
    def __init__(self):
        self.txt, self.ann = [], []
        self.X, self.y = [], []
        self.pos_samples = []

    def read(self, path):
        with ZipFile(path) as zf:
            all_txt = sorted([fn for fn in zf.namelist() if '.txt' in fn if fn[:20] == 'ROCstories/ROC_stori'])
            all_ann = sorted([fn for fn in zf.namelist() if '.ann' in fn if fn[:20] == 'ROCstories/ROC_stori'])
            for one_ann in all_ann:
                # print(one_ann)
                with zf.open(one_ann, 'r') as f:
                    # self.ann.append(f.read().encode())
                    ann = list(codecs.iterdecode(f, 'utf8'))
                    if len(ann) <= 7:
                        continue
                    self.ann.append(''.join(ann))
                with zf.open(one_ann[:-4] + '.txt', 'r') as f:
                    # self.txt.append(f.read().encode())
                    self.txt.append(''.join(list(codecs.iterdecode(f, 'utf8'))))
                # print()
        logging.info(f'{len(self.txt)} files loaded') 

    def preprocess(self, trial):
        processed = []
        for txt, ann in zip(self.txt, self.ann): # for every file
            # create tag and relation dictionaries
            T_dic, R_dic = {}, {}
            ann = [a.split('\t') for a in ann.split('\n')]
            for l in ann:
                if len(l) > 0 and l[0] != '':
                    if l[0][0] == 'T':
                        if ';' in l[1]:
                            all_idx = [y for x in l[1].split(';') for y in x.split()] # for a tag that span in differnt words not connected
                            T_dic[l[0]] = [all_idx[0], all_idx[1], all_idx[-1], l[2]]
                            print(l[1]) # TODO, not good fit?
                        else:
                            T_dic[l[0]] = l[1].split() + [l[2]]
                    elif l[0][0] == 'R':
                        l1 = l[1].split()
                        R_dic[l[0]] = [l1[0], re.sub(r'Arg\d:', '', l1[1]), re.sub(r'Arg\d:', '', l1[2])]

            sentences = nltk.sent_tokenize(txt)
            sentences_span = [[txt.find(sent), txt.find(sent) + len(sent)] for sent in sentences if len(sent) > 6]

            non_causal_sentences = list(range(len(sentences_span)))
            for rel in R_dic.values(): # every causal relation
                if 'CAUSE' in rel[0]:
                    arg1, arg2 = T_dic[rel[1]], T_dic[rel[2]]
                    arg1_span = [int(arg1[1]), int(arg1[2])]
                    arg2_span = [int(arg2[1]), int(arg2[2])]
                    args_span = [min(arg1_span + arg2_span), max(arg1_span + arg2_span)]
                    # print(arg1, arg2, args_span)

                    # this is for causal relation in one sentence
                    # want sent_span[0], args_span[0], args_span[1], sent_span[1]
                    # crpd_sent = [(sent, sent_span) for sent, sent_span in zip(sentences, sentences_span) if sent_span[0] <= args_span[0] <= args_span[1] <= sent_span[1]]
                    
                    sent_span0 = [i for i, sent_span in enumerate(sentences_span) if sent_span[0] <= args_span[0]][-1]
                    sent_span1 = [i for i, sent_span in enumerate(sentences_span) if args_span[1] <= sent_span[1]][0]
                    # print(sent_span0, sent_span1)
                    # remove those sentences are causal
                    for to_remove in range(sent_span0, sent_span1+1):
                        if to_remove in non_causal_sentences:
                            non_causal_sentences.remove(to_remove)
                    causal_sent = ' '.join(sentences[sent_span0:sent_span1+1])
                    # print(arg1[3], arg2[3])
                    # print(causal_sent)
                    # print(arg1[3] in causal_sent, arg2[3] in causal_sent)
                    # assert arg1[3] in causal_sent and arg2[3] in causal_sent, "didn't capture" # but doesn't work for span has gap
                    # causal_sent = ' '.join(sentences[sent_span0:sent_span1+1])
                    # print({'X': causal_sent, 'cause': arg1[3], 'effect': arg2[3]})
                    if '\n' in causal_sent:
                        assert len(causal_sent.split('\n')) <= 2, 'too many weired char in sent'
                        causal_sent = causal_sent.split('\n')[1]
                    processed.append({'X': causal_sent, 'causal': True, 'cause': arg1[3], 'effect': arg2[3]})
            for i in non_causal_sentences:
                non_causal_sent = re.sub(r"[*\n]", '', sentences[i]) # keep only good sentences # TODO not a universal fix
                processed.append({'X': non_causal_sent, 'causal': False})
                assert len(non_causal_sent) > 6
            if trial:
                break
        self.processed = pd.DataFrame(processed)
        # print(processed)

        logging.info(f'processed: {self.processed.shape}, causal: {self.processed.causal.sum()}')

    def save_output(self, data_path):
        utils.save_dt(self.processed, data_path, index=False)

    # def split(self, dev_prop=0.2, test_prop=0.2, seed=555):
    #     random.seed(seed)
    #     idx = list(range(len(self.X)))
    #     random.shuffle(idx)
    #     idx_1 = int(len(self.X) * (1-dev_prop-test_prop))
    #     idx_2 = idx_1 + int(len(self.X) * dev_prop)

    #     self.train_idx = idx[:idx_1]
    #     self.dev_idx = idx[idx_1:idx_2]
    #     self.test_idx = idx[idx_2:]
    #     logging.info(f'data splitted train: {len(self.train_idx)}, dev: {len(self.dev_idx)}, test: {len(self.test_idx)}')
    #     assert len(self.train_idx) + len(self.dev_idx) + len(self.test_idx) == len(self.X)

    # def write(self, data_path):
    #     with open(data_path, 'w+') as f:
    #         for i in self.train_idx:
    #             f.write(f'tr\t{self.y[i]}\t[CLS] {self.X[i]} [SEP]\n')
    #         for i in self.dev_idx:
    #             f.write(f'va\t{self.y[i]}\t[CLS] {self.X[i]} [SEP]\n')
    #         for i in self.test_idx:
    #             f.write(f'te\t{self.y[i]}\t[CLS] {self.X[i]} [SEP]\n')
    #     logging.info(f'data wrote')


# logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
# if __name__ == '__main__':
#     args = get_args()
#     # print(args)

#     dl = DataLoader()
#     dl.read(args.raw_data_path)
#     dl.preprocess()
#     dl.split()
#     dl.write(args.save_data_path + '/because_all.txt')
