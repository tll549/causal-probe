import re
import logging
import random
import numpy as np
import pandas as pd

class DataLoader(object):
    def __init__(self):
        self.X, self.y = [], []
        self.rel = []

    def read(self, data_path):
        with open(data_path) as f:
            next_is_y = False
            for line in f:
                if line[0].isnumeric():
                    self.X.append(line.split('\t')[1].split('\n')[0]) # only extract the sentence part
                    next_is_y = True
                else:
                    if next_is_y:
                        self.y.append(line.split('\n')[0])
                        next_is_y = False
        logging.debug(f'loaded X len: {len(self.X)}')

    def preprocess(self):
        all_relations_dict = {k:re.sub(r'\(e\d,e\d\)', '', k) for k in list(set(self.y))}

        X, y = [], []
        sub, obj = [], []
        for i in range(len(self.X)):
            self.rel.append(all_relations_dict[self.y[i]])

            e1_pattern, e2_pattern = r'<e1>(.*)</e1>', r'<e2>(.*)</e2>'
            e1 = re.findall(e1_pattern, self.X[i])[0]
            e2 = re.findall(e2_pattern, self.X[i])[0]
            sub_temp = e1 if '(e1,e2)' in self.y[i] else e2
            obj_temp = e2 if '(e1,e2)' in self.y[i] else e1
            sub.append(sub_temp)
            obj.append(obj_temp)
            x = re.sub(r'</?e\d>', '', self.X[i])[1:-1]
            X.append(x)

        self.X= X
        self.sub, self.obj = sub, obj

        logging.info(f'processed len X: {len(self.sub)}, causal: {sum([rel == "Cause-Effect" for rel in self.rel])}')

    def calc_prob(self, epsilon=1e-2, label='self'):
        def tokenize(X):
            '''split and lower'''
            handled_punct = [re.sub(r'([.,!?;])', r' \1', x) for x in X]
            return [[x2.lower() for x2 in x.split()] for x in handled_punct]
        tok = tokenize(self.X)
        num_words = len(set([w for sent in tok for w in sent]))

        logging.info(f'calculating features')
        self.output = pd.DataFrame(columns=[
            'X', 'relation', 'cause', 'effect', 
            'c_count', 'e_count', 'c_e_count', 'e_no_c_count', 'causal_dependency',
            'P(E|C)', 'P(E)', 'probabilistic_causality', 'probabilistic_causality_diff',
            'delta_P', 'P(E|no C)', 'q', 'p', 'causal_power'])
        for i in range(len(self.X)):
            c, e = self.sub[i],  self.obj[i]

            # causal dependency
            c_count = sum([c in sent for sent in self.X])
            e_count = sum([e in sent for sent in self.X])
            c_e_count = sum([c in sent and e in sent for sent in self.X])
            e_no_c_count = e_count - c_e_count # sum([e in sent and c not in sent for sent in self.X])
            causal_dependency = c_count == c_e_count and e_no_c_count == 0 # P(E|C) = 1 and P(E|not C) = 0

            # probabilistic causality
            P_of_E_given_C = c_e_count / c_count
            P_of_E = e_count / num_words
            probabilistic_causality = P_of_E_given_C >= P_of_E
            probabilistic_causality_diff = P_of_E_given_C - P_of_E

            # delta P
            P_E_given_no_C = e_no_c_count / (num_words - c_count)
            delta_P = P_of_E_given_C - P_E_given_no_C

            # causal power
            q = delta_P / (1 - P_E_given_no_C)
            if P_E_given_no_C == 0:
                # print('000000')
                p = 0
            else:
                p = -delta_P / P_E_given_no_C
            causal_power = q - p

            self.output.loc[i, :] = [self.X[i], self.rel[i], c, e, 
                c_count, e_count, c_e_count, e_no_c_count, causal_dependency,
                P_of_E_given_C, P_of_E, probabilistic_causality, probabilistic_causality_diff,
                delta_P, P_E_given_no_C, q, p, causal_power]
            # if i > 9:
            #     break 

        def MinMaxScaler(x):
            return (x - x.min()) / (x.max() - x.min())
        self.output['q_scaled'] = MinMaxScaler(self.output.q)
        self.output['p_scaled'] = MinMaxScaler(self.output.p)
        self.output['causal_power_scaled'] = MinMaxScaler(self.output.causal_power)

        logging.info(f'features calculated for {self.output.shape[0]} sentences')

    def save_output(self, data_path):
        self.output.to_csv(data_path, index=False)
        logging.info(f'output saved to {data_path}')

    # def split(self, dev_prop=0.2, test_prop=0.2, seed=555):
    #     random.seed(seed)
    #     idx = list(range(len(self.X)))
    #     random.shuffle(idx)
    #     idx_1 = int(len(self.X) * (1-dev_prop-test_prop))
    #     idx_2 = idx_1 + int(len(self.X) * dev_prop)

    #     self.train_idx = idx[:idx_1]
    #     self.dev_idx = idx[idx_1:idx_2]
    #     self.test_idx = idx[idx_2:]
    #     logging.debug(f'data splitted, train: {len(self.train_idx)}, dev: {len(self.dev_idx)}, test: {len(self.test_idx)}')
    #     assert len(self.train_idx) + len(self.dev_idx) + len(self.test_idx) == len(self.X)

    # def write(self, data_path):
    #     with open(data_path, 'w+') as f:
    #         for i in range(len(self.X)):
    #             f.write(f'te\t{self.y[i]}\t[CLS] {self.X[i]} [SEP]\t{self.rel[i]}\n')
    #     logging.info(f'data wrote')
