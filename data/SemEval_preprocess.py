import re
import logging
import random

import pandas as pd
from causal_probe import utils

class DataLoader(object):
    def __init__(self):
        self.X, self.y = [], []
        self.rel = []

    def read(self, data_path_both):
        for data_path in data_path_both:
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

    def preprocess(self, probing_task, mask='cause'):
        if probing_task == 'simple':
            # remove <e> tags
            self.X = [re.sub(r'</?e\d>', '', x) for x in self.X] # remove <e1>, </e1>, <e2>, </e2>
            assert all([x[0] == '"' for x in self.X] + [x[-1] == '"' for x in self.X])
            # remove " at the beginning and end
            self.X = [x[1:-1] for x in self.X]
            # retain y as only causal or not
            self.y = [int('Cause-Effect' in e) for e in self.y]

            self.output = pd.DataFrame()
            self.output['X'] = self.X
            self.output['causal'] = self.y
        if probing_task == 'mask':
            all_relations_dict = {k:re.sub(r'\(e\d,e\d\)', '', k) for k in list(set(self.y))}

            X, y, y2 = [], [], []
            for i in range(len(self.X)):
                self.rel.append(all_relations_dict[self.y[i]])
                # mask the cause or effect
                if (mask == 'cause' and '(e1,e2)' in self.y[i]) or \
                    (mask == 'effect' and '(e2,e1)' in self.y[i]):
                    pattern_to_mask = r'<e1>.*</e1>'
                    pattern_to_remove_tag = r'</?e2>'
                    pattern_y2 = r'<e2>.*</e2>'
                else:
                    pattern_to_mask = r'<e2>.*</e2>'
                    pattern_to_remove_tag = r'</?e1>'
                    pattern_y2 = r'<e1>.*</e1>'
                number_of_to_mask = len(re.findall(pattern_to_mask, self.X[i])[0].split(' '))
                masks = ' '.join(['[MASK]'] * number_of_to_mask)
                temp_X = re.sub(pattern_to_mask, masks, self.X[i])
                temp_X = re.sub(pattern_to_remove_tag, '', temp_X)[1:-1]
                X.append(temp_X)
                temp_y = re.findall(pattern_to_mask, self.X[i])[0]
                temp_y2 = re.findall(pattern_y2, self.X[i])[0]
                y.append(re.sub(r'</?e\d>', '', temp_y))
                y2.append(re.sub(r'</?e\d>', '', temp_y2))
            self.X, self.y, self.y2 = X, y, y2

        self.d = pd.DataFrame({'X': self.X, 'y': self.y, 'y2': self.y2})

        logging.info(f'processed len X: {len(self.X)}, causal: {sum([rel == "Cause-Effect" for rel in self.rel])}')

    def write(self, data_path):
        with open(data_path, 'w+') as f:
            for i in range(len(self.X)):
                f.write(f'te\t{self.y[i]}\t[CLS] {self.X[i]} [SEP]\t{self.rel[i]}\t{self.y2[i]}\n')
        logging.info(f'data wrote')

    def save_output(self, data_path):
        utils.save_dt(self.output, data_path, index=False)

# if __name__ == '__main__':
#     # args = get_args()
#     dl = DataLoader()
#     dl.read(args.raw_data_path)
#     dl.preprocess()
#     dl.split()
#     dl.write(args.save_data_path)