import re
import logging
# import argparse
import random

# def get_args():
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('-raw_data_path', type=str, 
#         default='data/causal_probing/SemEval_2010_8/raw/TRAIN_FILE.TXT',
#         help='raw data path to SemEval')
#     parser.add_argument('-save_data_path', type=str, 
#         default='data/causal_probing/SemEval_2010_8/processed/SemEval_processed.txt',
#         help='raw data path to SemEval')
#     return parser.parse_args()

class DataLoader(object):
    def __init__(self):
        self.X, self.y = [], []

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
                # if len(self.y) >= 741:
                #     break
        logging.debug(f'loaded X len: {len(self.X)}')

    def preprocess(self, probing_task, mask='cause'):
        if probing_task == 'simple':
            # remove <e> tags
            self.X = [re.sub(r'</?e\d>', '', x) for x in self.X] # remove <e1>, </e1>, <e2>, </e2>
            assert all([x[0] == '"' for x in self.X] + [x[-1] == '"' for x in self.X])
            # remove " at the beginning and end
            self.X = [x[1:-1] for x in self.X]
            # retain y as only causal or not
            self.y = [int('Cause-Effect' in e) for e in self.y]
        if probing_task == 'mask':
            X, y = [], []
            for i in range(len(self.X)):
                # keep only those are causal
                if 'Cause-Effect' not in self.y[i]:
                    continue
                # mask the cause or effect
                if (mask == 'cause' and self.y[i] == 'Cause-Effect(e1,e2)') or \
                    (mask == 'effect' and self.y[i] == 'Cause-Effect(e2,e1)'):
                    pattern_to_mask = r'<e1>.*</e1>'
                    pattern_to_remove_tag = r'</?e2>'
                else:
                    pattern_to_mask = r'<e2>.*</e2>'
                    pattern_to_remove_tag = r'</?e1>'
                number_of_to_mask = len(re.findall(pattern_to_mask, self.X[i])[0].split(' '))
                masks = ' '.join(['[MASK]'] * number_of_to_mask)
                temp_X = re.sub(pattern_to_mask, masks, self.X[i])
                X.append(re.sub(pattern_to_remove_tag, '', temp_X)[1:-1])
                temp_y = re.findall(pattern_to_mask, self.X[i])[0]
                y.append(re.sub(r'</?e\d>', '', temp_y))
            self.X, self.y = X, y

        logging.debug(f'processed len X: {len(self.X)}')

    def split(self, dev_prop=0.2, test_prop=0.2, seed=555):
        random.seed(seed)
        idx = list(range(len(self.X)))
        random.shuffle(idx)
        idx_1 = int(len(self.X) * (1-dev_prop-test_prop))
        idx_2 = idx_1 + int(len(self.X) * dev_prop)

        self.train_idx = idx[:idx_1]
        self.dev_idx = idx[idx_1:idx_2]
        self.test_idx = idx[idx_2:]
        logging.debug(f'data splitted, train: {len(self.train_idx)}, dev: {len(self.dev_idx)}, test: {len(self.test_idx)}')
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

# if __name__ == '__main__':
#     # args = get_args()
#     # print(args)

#     dl = DataLoader()
#     dl.read(args.raw_data_path)
#     dl.preprocess()
#     dl.split()
#     dl.write(args.save_data_path)
