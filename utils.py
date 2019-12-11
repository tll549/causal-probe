class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

import pandas as pd
from datetime import *
import logging

import pickle
def to_pickle_dt(var, filename, datetime_format="%Y%m%d%H%M"):
    datetime_now = datetime.now().strftime(datetime_format)
    filename_dt = filename + '_' + datetime_now + '.pkl'
    pickle.dump(var, open(filename_dt, 'wb'))
    logging.info(f'    done. {filename_dt} saved')

# def to_csv_dt(var, filename, verbose=1, datetime_format="%Y%m%d%H%M"):
#     datetime_now = datetime.now().strftime(datetime_format)
#     filename_dt = filename + '_' + datetime_now + '.csv'
#     if verbose:
#     	print(f'saving {filename}')
#     var.to_csv(filename_dt)
#     if verbose:
#         print(f'    done. {filename_dt} saved')

import os
import re
def find_newest(filename):
    if '/' in filename:
        filename_split = filename.split('/')
        path = '/'.join(filename_split[:-1]) + '/'
        filename_last = filename_split[-1]
    else:
        path = '.'
        filename_last = filename
    
#     filename_split = filename_last.split('.')
#     filename_last = '.'.join(filename_split[:-1])
#     suffix = '.' + filename_split[-1]

#     all_files = os.listdir(path)
#     all_matches = [x for x in all_files if re.match(fr'{filename_last}_\d{{12}}{suffix}', x)]
#     all_dt = [x[len(filename_last)+1:-len(suffix)] for x in all_matches]
#     assert all_dt != [], 'cannot find any files'

#     newest_dt = max(all_dt)
#     newest_filename = f'{filename_last}_{newest_dt}{suffix}'

#     if '/' in filename:
#         return path + newest_filename
#     else:
#         return newest_filename