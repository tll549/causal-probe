class dotdict(dict):
	""" dot.notation access to dictionary attributes """
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

import os
import re

import pandas as pd
from datetime import *
import logging

import pickle


class ProgressBar(object):
    def __init__(self, n_itr):
        self.n_itr = n_itr
    def now(self, i):
        if self.n_itr > 50:
            if i == 0:
                print(f'0%{"."*7}25%{"."*10}50%{"."*9}75%{"."*9}100%')
            elif (i+1) % round(self.n_itr/50) == 0:
                print('+', end = '')
            if i == self.n_itr - 1:
                print()



SUPPORTED_EXT = ['.pkl', '.csv', '.png', '.txt']

def sep_path_name_ext(path_name_ext):
	ext = [ext_test for ext_test in SUPPORTED_EXT if path_name_ext[-len(ext_test):] == ext_test]
	assert ext != [], 'please specify extension / extension not supported, currently support: {}'.format(SUPPORTED_EXT)
	ext = ext[0]
	path_name = path_name_ext[:-len(ext)]
	if '/' in path_name or '\\' in path_name:
		# splitted = path_name.split('/')
		splitted = re.split(r'/|\\', path_name)
		path = '/'.join(splitted[:-1])
		name = splitted[-1]
	else:
		path = ''
		name = path_name
	return path, name, ext

def append_dt(path_name_ext, datetime_format):
	path, name, ext = sep_path_name_ext(path_name_ext)

	datetime_now = datetime.now().strftime(datetime_format)
	if path != '':
		path_name_dt_ext = path + '/' + name + '_' + datetime_now + ext
	else:
		path_name_dt_ext = name + '_' + datetime_now + ext

	return path_name_dt_ext, ext

def save_dt(var, path_name_ext, datetime_format="%y%m%d%H%M", **kwargs):
	'''
	given 'path/path/filename.extension'
	save var to 'path/path/filename_datetime.entension'
	e.g., '../data/titanic.pkl'
	save var to '../data/titanic_201912091148.pkl'

	plt to '.png'

	for .txt
	var needs to be a list of string, each element is a line
	'''
	path_name_dt_ext, ext = append_dt(path_name_ext, datetime_format)

	if ext == '.pkl':
		with open(path_name_dt_ext, 'wb') as f:
			pickle.dump(var, f)

	elif ext == '.csv':
		assert isinstance(var, pd.DataFrame), 'not handled'
		var.to_csv(path_name_dt_ext, **kwargs)

	elif ext == '.png':
		if kwargs == {}: # default kwargs
			kwargs = {'dpi': 600, 'bbox_inches': 'tight'}

		var.savefig(path_name_dt_ext, **kwargs)
		
		try:
			var.close()
		except:
			# print("didn't close")
			pass

	elif ext == '.txt':
		with open(path_name_dt_ext, 'w+', encoding='utf-8', **kwargs) as f:
			f.writelines(var)

	else:
		assert False, 'currently can only handle' + str(SUPPORTED_EXT)

	logging.info(f'{path_name_dt_ext} saved')


def find_newest(path_name_ext):
	path, name, ext = sep_path_name_ext(path_name_ext)

	all_files = os.listdir(path)
	all_matches = [x for x in all_files if re.match(fr'{name}_\d{{10}}{ext}', x)]
	all_dt = [x[len(name)+1:-len(ext)] for x in all_matches]
	assert all_dt != [], 'cannot find any files, datetime must be 10 digits'

	newest_dt = max(all_dt)
	if path != '':
		newest_path_name_ext = path + '/' + name + '_' + newest_dt + ext
	else:
		newest_path_name_ext = name + '_' + newest_dt + ext
	return newest_path_name_ext, ext

def load_newest(path_name_ext):
	newest_path_name_ext, ext = find_newest(path_name_ext)
	if ext == '.pkl':
		with open(newest_path_name_ext, 'rb') as f:
			data = pickle.load(f)
	elif ext == '.csv':
		data = pd.read_csv(newest_path_name_ext)
	else:
		print('not handled')

	logging.info(f'{newest_path_name_ext} loaded')
	return data

explicit_kw = ['cause', 'caused', 'causes', 'causing', 
               'result', 'resulted', 'results', 'resulting',
               'due', # prepositions
               'because', 'since', 'so that', # conjunctions
               'so', 'therefore', 'thus', # adverbial connectors
               'the reason why', 'the result is', 'that is why', # clause-integrated expressions
               'lead to' # mine
              ]
def detect_kw(tok, kw_list=explicit_kw):
    '''detect if ['ab', 'bc', 'cd'], ['bc', 'ef']'''
    for w in tok:
        for kw in kw_list:
            if kw == w:
                return True
            if ' ' in kw:
                if kw in ' '.join(tok):
                    return True
    return False