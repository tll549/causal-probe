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
import numpy as np

from nltk.probability import FreqDist
from nltk.util import bigrams, trigrams

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
        self.output = pd.DataFrame(processed)
        logging.info(f'processed: {self.output.shape}, causal: {self.output.causal.sum()}')
        self.X, self.sub, self.obj = self.output.X, self.output.cause,  self.output.effect

    def calc_prob(self):
        def tokenize(X):
            '''split and lower'''
            handled_punct = [re.sub(r'([.,!?;])', r' \1', x) for x in X]
            return [[x2.lower() for x2 in x.split()] for x in handled_punct]
        tok = tokenize(self.X)
        # num_words = len(set([w for sent in tok for w in sent])) # can't use this, causes negative in prob causality diff because P(E|C) is counting by sentences, not by words
        self.num_sent = len(self.X)

        logging.info(f'calculating features...')
        self.output = pd.DataFrame(columns=[
            'X', 'relation', 'cause', 'effect', 
            'c_count', 'e_count', 'c_e_count', 'e_no_c_count', 'causal_dependency',
            'P(E|C)', 'P(E)', 'probabilistic_causality', 'probabilistic_causality_diff',
            'delta_P', 'P(E|no C)', 'q', 'p', 'causal_power'])
        for i in range(len(self.X)):
            c, e = self.sub[i],  self.obj[i]
            if c != c or e != e: # nan, not causal
                continue

            # causal dependency
            c_count = sum([c in sent for sent in self.X])
            e_count = sum([e in sent for sent in self.X])
            c_e_count = sum([c in sent and e in sent for sent in self.X])
            e_no_c_count = e_count - c_e_count # sum([e in sent and c not in sent for sent in self.X])
            causal_dependency = c_count == c_e_count and e_no_c_count == 0 # P(E|C) = 1 and P(E|not C) = 0

            # probabilistic causality
            P_of_E_given_C = c_e_count / c_count if c_count != 0 else np.nan
            P_of_E = e_count / self.num_sent
            probabilistic_causality = P_of_E_given_C >= P_of_E
            probabilistic_causality_diff = P_of_E_given_C - P_of_E
            # assert probabilistic_causality_diff > 0, f'{P_of_E_given_C}, {P_of_E}, {c_e_count}, {c_count}, {e_count}'

            # delta P
            P_E_given_no_C = e_no_c_count / (self.num_sent - c_count)
            delta_P = P_of_E_given_C - P_E_given_no_C

            # causal power
            q = delta_P / (1 - P_E_given_no_C)
            if P_E_given_no_C == 0:
                # print('000000')
                p = 0
            else:
                p = -delta_P / P_E_given_no_C
            causal_power = q - p

            self.output.loc[i, :] = [self.X[i], 'Cause-Effect', c, e, 
                c_count, e_count, c_e_count, e_no_c_count, causal_dependency,
                P_of_E_given_C, P_of_E, probabilistic_causality, probabilistic_causality_diff,
                delta_P, P_E_given_no_C, q, p, causal_power]
            # if i > 10:
            #     break 
        logging.info(f'features calculated for {self.output.shape[0]} sentences')


    def calc_prob_oanc(self, oanc_datapath, use_semeval_first=True, trial=False):

        if use_semeval_first: # can avoid c_count = 0 in oanc
            self.calc_prob()
        else:
            self.output = pd.DataFrame({'X': self.X, 'relation': self.rel, 'cause': self.sub, 'effect': self.obj})
            for c in ['c_count', 'e_count', 'c_e_count', 'e_no_c_count']:
                self.output[c] = 0
            self.num_sent = 0

        logging.info(f'calculating features using OANC...')
        fdist_uni, fdist_bi, fdist_tri = FreqDist(), FreqDist(), FreqDist()
        with ZipFile(oanc_datapath) as zf:
            txt_written_files = [fn for fn in zf.namelist() if '.txt' in fn and 'written' in fn]
            logging.info(f'there are {len(txt_written_files)} txt written files')
            pb = utils.ProgressBar(len(txt_written_files))
            for f_i, f_path in enumerate(txt_written_files):
                # print(f_path)
                pb.now(f_i)
                all_lines = ''
                with zf.open(f_path) as f:
                    for line in codecs.iterdecode(f, 'utf8'):
                        all_lines += line
                all_lines = re.sub(r'\s\s+', ' ', all_lines) # remove repeated spaces ' +'
                
                # sep and process by line
                sentences = nltk.sent_tokenize(all_lines)
                for sent in sentences:
                    tok = [w.lower() for w in word_tokenize(sent)]
                    self.num_sent += 1
                    # print(tok)
                    c_in = self.output.cause.isin(tok)
                    e_in = self.output.effect.isin(tok)
                    self.output.c_count += c_in
                    self.output.e_count += e_in
                    self.output.c_e_count += c_in & e_in
                    self.output.e_no_c_count += ~c_in & e_in

                # unigram, bigram, and trigram for calc avg freq
                for word in tok:
                    fdist_uni[word] += 1
                for bi in bigrams(tok):
                    fdist_bi[bi] += 1
                for tri in trigrams(tok):
                    fdist_tri[bi] += 1

                if trial:
                    if f_i > 10:
                        break
        logging.info(f'iterated through OANC, {self.num_sent} sentences')

        self.output.c_count = self.output.c_count.astype(int)
        self.output.e_count = self.output.e_count.astype(int)
        self.output.c_e_count = self.output.c_e_count.astype(int)
        self.output.e_no_c_count = self.output.e_no_c_count.astype(int)

        # causal dependency
        self.output['causal_dependency'] = (self.output.c_count == self.output.c_e_count) & (self.output.e_no_c_count == 0)
        # probabilistic causality
        self.output['P(E|C)'] = self.output.c_e_count / self.output.c_count
        self.output['P(E)'] = self.output.e_count / self.num_sent
        self.output['probabilistic_causality'] = self.output['P(E|C)'] >= self.output['P(E)']
        self.output['probabilistic_causality_diff'] = self.output['P(E|C)'] - self.output['P(E)']
        # delta P
        self.output['P(E|no C)'] = self.output.e_no_c_count / (self.num_sent - self.output.c_count)
        self.output['delta_P'] = self.output['P(E|C)'] - self.output['P(E|no C)']

        # causal power
        self.output['q'] = self.output.delta_P / (1 - self.output['P(E|no C)'])
        self.output['p'] = (-self.output.delta_P / self.output['P(E|no C)'].replace({0 : np.nan})).fillna(0) # handle divide by 0
        self.output['causal_power'] = self.output.q - self.output.p

        # PMI
        # D11-1027, Do et al., 2017
        self.N = fdist_uni.N()
        self.output['PMI'] = np.log(self.output.c_e_count.astype(int) * self.N / (self.output.c_count.astype(int) * self.output.e_count.astype(int))) # don't know why dtype is obj

        # PPMI, CPMI, NPMI, NNEGPMI
        # Salle & Villavicencio, 2019
        self.output.loc[self.output.PMI >= 0, 'PPMI'] = self.output.loc[self.output.PMI >= 0, 'PMI']
        self.output.loc[self.output.PMI < 0, 'PPMI'] = 0

        self.output.loc[self.output.PMI >= -2, 'CPMI_-2'] = self.output.loc[self.output.PMI >= -2, 'PMI']
        self.output.loc[self.output.PMI < -2, 'CPMI_-2'] = -2

        self.output['NPMI'] = self.output['PMI'] / -np.log(self.output.c_e_count.astype(int) / self.N)

        self.output.loc[self.output['PMI'] >= 0, 'NNEGPMI'] = self.output.loc[self.output['PMI'] >= 0, 'PMI']
        self.output.loc[self.output['PMI'] < 0, 'NNEGPMI'] = self.output.loc[self.output['PMI'] < 0, 'NPMI']

        # causal strength
        # Luo et al., 2016
        alpha = 0.66
        self.output['P(C|E)'] = self.output.c_e_count / self.output.e_count
        self.output['causal_stength_nec'] = (self.output['P(C|E)'] / self.N) / (self.output.c_count / self.N) ** alpha
        self.output['causal_stength_suf'] = (self.output['P(E|C)'] / self.N) / (self.output.e_count / self.N) ** alpha
        lambda_cs_list = [0.5, 0.7, 0.9, 1.0]
        for lambda_cs in lambda_cs_list:
            self.output[f'causal_stength_{lambda_cs}'] = self.output.causal_stength_nec ** lambda_cs * self.output.causal_stength_suf ** (1 - lambda_cs)

        # avg frequency, overall frequency
        def calc_avg_freq(s, fdist):
            return np.mean([fdist[w] for w in s]) / fdist.N()
        def calc_ovr_freq(s, fdist):
            return np.sum(np.log([fdist[w] / fdist.N() for w in s]))
        X_unigram = self.output.X.apply(lambda x: [w.lower() for w in word_tokenize(x)])
        X_bigram = X_unigram.apply(lambda x: list(bigrams(x)))
        X_trigram = X_unigram.apply(lambda x: list(trigrams(x)))
        self.output['avg_freq_uni'] = X_unigram.apply(calc_avg_freq, args=(fdist_uni, ))
        self.output['avg_freq_bi'] = X_bigram.apply(calc_avg_freq, args=(fdist_bi, ))
        self.output['avg_freq_tri'] = X_trigram.apply(calc_avg_freq, args=(fdist_tri, ))
        self.output['ovr_freq_uni'] = X_unigram.apply(calc_ovr_freq, args=(fdist_uni, ))
        self.output['ovr_freq_bi'] = X_bigram.apply(calc_ovr_freq, args=(fdist_bi, ))
        self.output['ovr_freq_tri'] = X_trigram.apply(calc_ovr_freq, args=(fdist_tri, ))

        if trial:
            pd.set_option('display.max_columns', 1000)
            print(self.output.head())

    def make_categorical(self, num_classes, num_classes_by):
        '''make each numerical variables in each relation categorical'''
        def float_categorize(s, num_classes, by):
            '''s should be numerical pd.series'''
            if by == 'linear':  
                s = ((s - s.min()) / ((s.max() - s.min()) / num_classes)).astype(int)
                s[s == num_classes] = num_classes - 1
                return s
            elif by == 'quantile':
                return pd.qcut(s, num_classes, labels=False)
        numerical_columns = [c for c in self.output.columns[8:] if self.output[c].nunique() > num_classes]
        for c in numerical_columns:
            for rel in self.output.relation.unique():
                try:
                    self.output.loc[self.output.relation==rel, c + '_cat'] = \
                        float_categorize(pd.to_numeric(self.output.loc[self.output.relation==rel, c]), 
                            num_classes, num_classes_by)
                except ValueError:
                    print(f'{c}, {rel}, this combination will be dropped later, ValueError: Bin edges must be unique')
                except KeyError:
                    print(f'{c}, {rel} KeyError')
                # assert self.output[c].nunique() <= num_classes, f'more than {num_classes} classes'
        # remove incomplete cases caused from above
        # self.output = self.output.dropna()

    def save_output(self, data_path):
        utils.save_dt(self.output, data_path, index=False)