import re
import logging
import random

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
            all_relations_dict = {k:re.sub(r'\(e\d,e\d\)', '', k) for k in list(set(self.y))}

            X, y = [], []
            for i in range(len(self.X)):
                self.rel.append(all_relations_dict[self.y[i]])
                # mask the cause or effect
                if (mask == 'cause' and '(e1,e2)' in self.y[i]) or \
                    (mask == 'effect' and '(e2,e1)' in self.y[i]):
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

        logging.info(f'processed len X: {len(self.X)}, causal: {sum([rel == "Cause-Effect" for rel in self.rel])}')

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
            for i in range(len(self.X)):
                f.write(f'te\t{self.y[i]}\t[CLS] {self.X[i]} [SEP]\t{self.rel[i]}\n')
        logging.info(f'data wrote')



# logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# if __name__ == '__main__':
#     # args = get_args()
#     # print(args)

#     dl = DataLoader()
#     dl.read(args.raw_data_path)
#     dl.preprocess()
#     dl.split()
#     dl.write(args.save_data_path)
