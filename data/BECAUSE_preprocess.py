import re
import logging
import argparse
import random

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-raw_data_path', type=str, 
        default='data/causal_probing/BECAUSE/NYT/1818253.ann',
        help='raw data path to SemEval')
    parser.add_argument('-save_data_path', type=str, 
        default='data/causal_probing/BECAUSE/processed/BECAUSE_NYT_processed.txt',
        help='raw data path to SemEval')
    return parser.parse_args()

class DataLoader(object):
    def __init__(self):
        self.X, self.y = [], []

        # self.path = path
        self.text, self.idx, self.t_dic, self.e_dic, self.temp = [], set(), {}, {}, {}
        self.train = {'idx': [], 't_dic': {}, 'e_dic': {}, 'temp': {}}
        self.dev = {'idx': [], 't_dic': {}, 'e_dic': {}, 'temp': {}}
        self.test = {'idx': [], 't_dic': {}, 'e_dic': {}, 'temp': {}}
        # self.read_annot(path)

    def read(self, path):
        raw_text, new_text, raw_index, corr_index, raw_length, sent = [], [], [], [], 0, ''
        with open(path, "r") as raw_fp:
            new_index = len(self.text) - 1
            for line in raw_fp:
                print(line)
                raw_text.append(line)
                sent_len = len(line)
                raw_index.append(raw_length)
                raw_length += sent_len
                marker = len(line) - len(line.lstrip('\t'))
                new_line = line.strip('\n').strip('\t')
                print(new_line)
                if new_line:
                    if marker == 3:
                        new_text.append(sent)
                        new_index += 1
                        corr_index.append(new_index)
                        sent = new_line
                    elif marker == 4:
                        corr_index.append(new_index)
                        sent += ' ' + new_line
                    else:
                        print(marker)
                else:
                    corr_index.append(-1)
                break
        new_text.append(sent)
        self.text += new_text[1:]
        return raw_text, raw_index, corr_index


    def preprocess(self):
        # remove <e> tags
        self.X = [re.sub(r'</?e\d>', '', x) for x in self.X] # remove <e1>, </e1>, <e2>, </e2>
        assert all([x[0] == '"' for x in self.X] + [x[-1] == '"' for x in self.X])
        # remove " at the beginning and end
        self.X = [x[1:-1] for x in self.X]
        # retain y as only causal or not
        self.y = [int('Cause-Effect' in e) for e in self.y]

        logging.info(f'number of y == 1: {sum(self.y)}, 0: {len(self.y)-sum(self.y)}')

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


logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    args = get_args()
    # print(args)

    dl = DataLoader()
    dl.read(args.raw_data_path)
    # dl.preprocess()
    # dl.split()
    # dl.write(args.save_data_path)
