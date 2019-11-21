'''
https://github.com/lingyugao/causal/blob/master/data.py
'''

import os
import math
import glob
import nltk
import torch
import random
import logging
import itertools
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize


# nltk.data.path.append(os.path.expanduser('~/data/origin/nltk_data'))

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DataLoader(object):
    def __init__(self, path):
        self.path = path
        self.text, self.idx, self.t_dic, self.e_dic, self.temp = [], set(), {}, {}, {}
        self.train = {'idx': [], 't_dic': {}, 'e_dic': {}, 'temp': {}}
        self.dev = {'idx': [], 't_dic': {}, 'e_dic': {}, 'temp': {}}
        self.test = {'idx': [], 't_dic': {}, 'e_dic': {}, 'temp': {}}
        self.read_annot(path)

    def read_raw_text(self, path):
        """
        get the full sentence as the sentences in raw text are separated by '\t' and '\n'
        3'\t'means start and 4 '\t'means part of previuos sentence.
        use number of '\t' to mark the sentence ending
        :param path: path to raw text
        :return: raw text and character index, new processed text and character index
        """
        raw_text, new_text, raw_index, corr_index, raw_length, sent = [], [], [], [], 0, ''
        with open(path, "r") as raw_fp:
            new_index = len(self.text) - 1
            for line in raw_fp:
                raw_text.append(line)
                sent_len = len(line)
                raw_index.append(raw_length)
                raw_length += sent_len
                marker = len(line) - len(line.lstrip('\t'))
                new_line = line.strip('\n').strip('\t')
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
        new_text.append(sent)
        self.text += new_text[1:]
        return raw_text, raw_index, corr_index

    def read_annot(self, path):
        """
        read annotation data, match and verify
        :param path: path of annotation file
        t_dic: label(e.g. T1): (sentence_index, start point, length)
        e_dic: label(e.g. E1): (e_pos, e_class, e_marker, e_cause, e_effect)
        """
        raw_text, raw_index, corr_index = self.read_raw_text(path.rsplit('.', 1)[0] + '.txt')
        with open(path, 'r') as ann_fp:
            pointer = 0
            for line in ann_fp:
                if line[0] == 'T':
                    t_label, t_char, t_phrase = line.strip('\n').split('\t')
                    if ';' in t_char:
                        t_type, *t_raw_pos = t_char.replace(';', ' ').split(' ')
                        t_raw_start = int(t_raw_pos[0])
                        t_raw_end = int(t_raw_pos[-1])
                    else:
                        t_type, t_raw_start, t_raw_end = t_char.split(' ')
                    t_raw_start, t_raw_end = int(t_raw_start), int(t_raw_end)

                    # find sentence position in document
                    if t_raw_start >= raw_index[pointer]:
                        while t_raw_start >= raw_index[pointer]:
                            pointer += 1
                        pointer -= 1
                    else:
                        pointer = 0
                        while t_raw_start >= raw_index[pointer]:
                            pointer += 1
                        pointer -= 1

                    # check phrase position
                    t_pos = corr_index[pointer]
                    t_start = self.text[t_pos].find(t_phrase)
                    if t_start == -1:
                        # the phrase is separated into at least two parts
                        p_raw_start = 0
                        t_start, t_len = [], []
                        for i in range(int(len(t_raw_pos)/2)):
                            p_len = int(t_raw_pos[i+1]) - int(t_raw_pos[i])
                            if len(t_phrase) >= p_raw_start + p_len and t_phrase[p_raw_start + p_len].isalpha():
                                if t_phrase[p_raw_start] == ' ':
                                    p_raw_start += 1
                                else:
                                    raise ValueError('Check your input!')
                            p_start = self.text[t_pos].find(t_phrase[p_raw_start:p_raw_start + p_len].strip())
                            assert(p_start != -1)
                            t_start.append(p_start)
                            t_len.append(p_len)
                            p_raw_start += p_len
                        self.t_dic[t_label] = (t_pos, t_start, t_len, t_phrase)
                    else:
                        self.t_dic[t_label] = (t_pos, t_start, len(t_phrase), t_phrase)
                    self.idx.add(t_pos)
                elif line[0] == 'E':
                    e_label, *e_phrase = line.replace('\t', ' ').strip('\n').split(' ')
                    e_phrase = [x.split(':') for x in e_phrase]
                    e_pos = self.t_dic[e_phrase[0][1]][0]
                    e_class, e_marker = e_phrase[0]
                    e_cause, e_effect = '', ''
                    for label, event in e_phrase[1:]:
                        if label in ['Cause', 'Arg0']:
                            e_cause = event
                        elif label in ['Effect', 'Arg1']:
                            e_effect = event
                        else:
                            print(e_class)
                        # assert(t_dic[event][0] == e_pos)    # relations are in the same sentence
                    self.e_dic[e_label] = (e_pos, e_class, e_marker, e_cause, e_effect)
                elif line[0] == 'A':
                    a_label, *a_phrase = line.replace('\t', ' ').strip('\n').split(' ')
                    if a_phrase[0] == 'Temporal':
                        assert(len(a_phrase) == 2)
                        event = a_phrase[1]
                        self.temp[a_label] = (self.e_dic[event][0], event)
                elif line[0] == 'R':
                    continue
                    # e_label, *e_phrase = line.replace('\t', ' ').strip('\n').split(' ')
                elif line[0] == '#':
                    continue
                else:
                    raise ValueError('check the annotation file!')

    # split data to train, dev and test
    def split(self):
        full_size = len(self.text)
        real_size = len(self.idx)
        logging.info('sentence number: {:d}'.format(full_size))
        logging.info('useful sentence number: {:d}'.format(real_size))
        dev_size = int(math.floor(full_size / 4))
        train_size = full_size - 2 * dev_size
        ref_list = list(range(full_size))
        random.shuffle(ref_list)
        self.train['idx'], self.dev['idx'], self.test['idx'] = ref_list[:train_size], \
                                                               ref_list[train_size:train_size + dev_size], \
                                                               ref_list[train_size + dev_size:]
        dict_list = ['t_dic', 'e_dic', 'temp']
        for idx, full_dict in enumerate([self.t_dic, self.e_dic, self.temp]):
            for key in full_dict:
                key_idx = ref_list.index(full_dict[key][0])
                if key_idx < train_size:
                    self.train[dict_list[idx]][key] = full_dict[key]
                elif key_idx < train_size + dev_size and key_idx >= train_size:
                    self.dev[dict_list[idx]][key] = full_dict[key]
                elif key_idx >= train_size + dev_size:
                    self.test[dict_list[idx]][key] = full_dict[key]


class NYTDataLoader(DataLoader):
    def __init__(self, path):
        super(NYTDataLoader, self).__init__(path)

    def read_raw_text(self, path):
        """
        get the full sentence as the sentences in raw text are separated by '\t' and '\n'
        3'\t'means start and 4 '\t'means part of previuos sentence.
        use number of '\t' to mark the sentence ending
        :param path: path to raw text
        :return: raw text and character index, new processed text and character index
        """
        raw_text, new_text, raw_index, corr_index, raw_length, sent = [], [], [], [], 0, ''
        with open(path, "r") as raw_fp:
            new_index = len(self.text) - 1
            for line in raw_fp:
                raw_text.append(line)
                new_line = line.strip('\n').strip('\t')
                if new_line:
                    sent_len = len(new_line)
                    raw_index.append(raw_length)
                    raw_length += sent_len
                    new_text.append(sent)
                    new_index += 1
                    corr_index.append(new_index)
                    sent = new_line
        new_text.append(sent)
        self.text = new_text[1:]
        return raw_text, raw_index, corr_index

    # def read_raw_text(self, path):
    #     """
    #     get the full sentence as the sentences in raw text are separated by '\t' and '\n'
    #     3'\t'means start and 4 '\t'means part of previuos sentence.
    #     use number of '\t' to mark the sentence ending
    #     :param path: path to raw text
    #     :return: raw text and character index, new processed text and character index
    #     """
    #     raw_text, new_text, raw_index, corr_index, raw_length, sent = [], [], [], [], 0, ''
    #     with open(path, "r") as raw_fp:
    #         new_index = len(self.text)
    #         for line in raw_fp:
    #             raw_text.append(line)
    #             if line.strip('\n').strip('\t'):
    #                 tokenized_line = sent_tokenize(line)
    #                 for sentence in tokenized_line:
    #                     sent_len = len(sentence)
    #                     raw_index.append(raw_length)
    #                     raw_length += sent_len
    #                 if len(tokenized_line) > 1:
    #                     for i in range(1, len(tokenized_line)):
    #                         raw_index[-i] += 1
    #                         raw_length += 1
    #                 new_text += tokenized_line
    #                 corr_index += list(range(new_index, new_index + len(tokenized_line)))
    #                 new_index += len(tokenized_line)
    #             else:
    #                 # corr_index.append(-1)
    #                 raw_length += 1
    #     for sentence in new_text:
    #         self.text.append(sentence.replace("``", "\'\'"))
    #     # raw_index[0] -= 1
    #     return raw_text, raw_index, corr_index

    def read_annot(self, path):
        """
        read annotation data, match and verify
        :param path: path of annotation file
        t_dic: label(e.g. T1): (sentence_index, start point, length)
        e_dic: label(e.g. E1): (e_pos, e_class, e_marker, e_cause, e_effect)
        """
        raw_text, raw_index, corr_index = self.read_raw_text(path.rsplit('.', 1)[0] + '.txt')
        with open(path, 'r') as ann_fp:
            pointer = 0
            for line in ann_fp:
                if line[0] == 'T':
                    t_label, t_char, t_phrase = line.strip('\n').split('\t')
                    if ';' in t_char:
                        t_type, *t_raw_pos = t_char.replace(';', ' ').split(' ')
                        t_raw_start = int(t_raw_pos[0])
                        t_raw_end = int(t_raw_pos[-1])
                    else:
                        t_type, t_raw_start, t_raw_end = t_char.split(' ')
                    t_raw_start, t_raw_end = int(t_raw_start), int(t_raw_end)

                    # find sentence position in document
                    if t_raw_start >= raw_index[pointer]:
                        while t_raw_start >= raw_index[pointer]:
                            pointer += 1
                            if pointer == len(raw_index):
                                break
                        pointer -= 1
                    else:
                        pointer = 0
                        while t_raw_start >= raw_index[pointer]:
                            pointer += 1
                            if pointer == len(raw_index):
                                break
                        pointer -= 1

                    # check phrase position
                    t_pos = corr_index[pointer]
                    t_start = self.text[t_pos].find(t_phrase)
                    if t_start == -1:
                        new_text = self.text[t_pos].replace("``", "\'\'").replace("''", "\'\'")
                        new_phrase = t_phrase.replace("``", "\'\'").replace("''", "\'\'")
                        t_start = new_text.find(new_phrase)
                        if t_start == -1:
                            # the phrase is separated into at least two parts
                            p_raw_start = 0
                            t_start, t_len = [], []
                            try:
                                t_raw_pos
                            except NameError:
                                t_start = self.text[t_pos - 1].find(t_phrase)
                                print('check')
                                self.t_dic[t_label] = (t_pos - 1, t_start, t_len, t_phrase)
                            else:
                                for i in range(int(len(t_raw_pos) / 2)):
                                    p_len = int(t_raw_pos[2 * i + 1]) - int(t_raw_pos[2 * i])
                                    if len(t_phrase) > p_raw_start + p_len:
                                        if t_phrase[p_raw_start + p_len].isalpha():
                                            if t_phrase[p_raw_start] == ' ':
                                                p_raw_start += 1
                                    elif len(t_phrase) == p_raw_start + p_len:
                                        print('check')
                                    p_start = self.text[t_pos].find(t_phrase[p_raw_start: p_raw_start + p_len].
                                                                    strip())
                                    if p_start == -1:
                                        p_start = self.text[t_pos - 1].find(t_phrase[p_raw_start: p_raw_start + p_len].
                                                                        strip())
                                        assert (p_start != -1)
                                    t_start.append(p_start)
                                    t_len.append(p_len)
                                    p_raw_start += p_len
                                self.t_dic[t_label] = (t_pos, t_start, t_len, t_phrase)
                    else:
                        self.t_dic[t_label] = (t_pos, t_start, len(t_phrase), t_phrase)
                    self.idx.add(t_pos)
                elif line[0] == 'E':
                    e_label, *e_phrase = line.replace('\t', ' ').strip('\n').split(' ')
                    e_phrase = [x.split(':') for x in e_phrase]
                    e_pos = self.t_dic[e_phrase[0][1]][0]
                    e_class, e_marker = e_phrase[0]
                    e_cause, e_effect = '', ''
                    for label, event in e_phrase[1:]:
                        if label in ['Cause', 'Arg0']:
                            e_cause = event
                        elif label in ['Effect', 'Arg1']:
                            e_effect = event
                        else:
                            print(e_class)
                        # assert(t_dic[event][0] == e_pos)    # relations are in the same sentence
                    self.e_dic[e_label] = (e_pos, e_class, e_marker, e_cause, e_effect)
                elif line[0] == 'A':
                    a_label, *a_phrase = line.replace('\t', ' ').strip('\n').split(' ')
                    if a_phrase[0] == 'Temporal':
                        assert (len(a_phrase) == 2)
                        event = a_phrase[1]
                        self.temp[a_label] = (self.e_dic[event][0], event)
                elif line[0] == 'R':
                    continue
                    # e_label, *e_phrase = line.replace('\t', ' ').strip('\n').split(' ')
                elif line[0] == '#':
                    continue
                else:
                    raise ValueError('check the annotation file!')


class PTBDataLoader(DataLoader):
    def __init__(self, path):
        super(PTBDataLoader, self).__init__(path)

    def split(self):
        return


class HearingDataLoader(DataLoader):
    def __init__(self, path):
        super(HearingDataLoader, self).__init__(path)

    def split(self):
        return


class MASCDataLoader(DataLoader):
    def __init__(self, path):
        super(MASCDataLoader, self).__init__(path)

    def split(self):
        return


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

    dl = DataLoader(os.path.expanduser('data/causal_probing/BECAUSE/MASC'))
    dl.get_file()
    dl.split()



# import os
# import glob
# import logging
# from shutil import copyfile

# output_path = os.path.expanduser('data/causal_probing/BECAUSE/NYT')
# nyt_path = os.path.expanduser('data/causal_probing/BECAUSE/NYT')

# ann = glob.glob(os.path.join(output_path, '*.ann'))
# text = glob.glob(os.path.join(nyt_path, '*.txt'))
# # ann = glob.glob('data/causal_probing/BECAUSE/NYT/*.ann')
# # out = glob.glob('data/causal_probing/BECAUSE/NYT/*.ann')
# for file in ann:
#     print(file)
#     substring = file.rsplit('.', 1)[0] + '.txt'
#     print(substring)
#     if substring in text:
#         print(substring)
# #         res = [i for i in text if substring in i]
# #         assert(len(res) == 1)
# #         file_name = os.path.splitext(os.path.basename(file))[0]
# #         logging.info('read data: ' + file_name)
# #         copyfile(res[0], os.path.join(output_path, file_name + '.txt'))
#     else:
#         logging.info('error! {}'.format(file))
#     break