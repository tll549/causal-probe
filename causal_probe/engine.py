import io
import numpy as np
import pandas as pd
import logging
import os
import re

from causal_probe import utils
from data import SemEval_preprocess
from data import SemEval_feature_preprocess
from data import ROC_preprocess
from data import BECAUSE_preprocess

from nltk.tokenize import word_tokenize

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import GPT2Tokenizer, GPT2Model

from causal_probe.classifier import MLP

import matplotlib.pyplot as plt
import seaborn as sns

GLOVE_PATH = 'examples/glove/glove.840B.300d.txt'
CONCEPTNET_PATH = 'examples/conceptnet/numberbatch-en-19.08.txt.gz'

DATAPATH = 'data/causal_probing/'

SEMEVAL_PATH = 'SemEval_2010_8'
SEMEVAL_DATA = ['TRAIN_FILE.TXT', 'TEST_FILE_FULL.TXT'] # 'data/causal_probing/SemEval_2010_8/raw/TRAIN_FILE.TXT'

ROC_PATH = 'ROCStories'
ROC_DATA = 'ROCstories-20191212T222034Z-001.zip' # 'data/causal_probing/ROCStories/ROCstories-20191212T222034Z-001.zip'

BECAUSE_PATH = 'BECAUSE'
# BECAUSE_DATA path is specify in BECAUSE_preprocess

OANC_DATA = 'OANC_GrAF.zip' # 'data/causal_probing/OANC_GrAF.zip'

LOGS_PATH = 'logs/'

SemEval_feature_PROCESSED_DATAPATH = 'data/causal_probing/SemEval_2010_8/processed/SemEval_feature_processed.csv'
SemEval_feature_ENCODED_DATAPATH = 'data/causal_probing/SemEval_2010_8/processed/SemEval_feature_embeddings.pkl'

ROC_RAW_DATAPATH = 'data/causal_probing/ROCStories/ROCstories-20191212T222034Z-001.zip'
ROC_feature_PROCESSED_DATAPATH = 'data/causal_probing/ROCStories/processed/ROC_feature_processed.csv'
ROC_feature_ENCODED_DATAPATH = 'data/causal_probing/ROCStories/processed/ROC_feature_embeddings.pkl'

OANC_DATAPATH = 'data/causal_probing/OANC_GrAF.zip'

class engine(object):
	def __init__(self, params):
		self.params = utils.dotdict(params)
		self.params.pretrained = utils.dotdict(self.params.pretrained)
		logging.info('params: ' + str(self.params))

		# raw data path
		if self.params.dataset == 'semeval':
			self.raw_datapath = [os.path.join(DATAPATH, SEMEVAL_PATH, 'raw', SEMEVAL_DATA[0]),
				os.path.join(DATAPATH, SEMEVAL_PATH, 'raw', SEMEVAL_DATA[1])]
		elif self.params.dataset == 'because':
			self.raw_datapath = os.path.join(DATAPATH, BECAUSE_PATH)
		elif self.params.dataset == 'roc':
			self.raw_datapath = os.path.join(DATAPATH, ROC_PATH, ROC_DATA)

		# processed data path, encoded data path, result csv data path, and fig data path
		if self.params.probing_task == 'simple':
			config_filename = ('TRIAL_' if self.params.trial else '') + \
				'_'.join([self.params.probing_task, self.params.dataset, str(self.params.seed)])
			if self.params.dataset == 'semeval':
				dataset_path = SEMEVAL_PATH
			elif self.params.dataset == 'because':
				dataset_path = BECAUSE_PATH
			elif self.params.dataset == 'roc':
				dataset_path = ROC_PATH

			# universal setting?
			self.processed_datapath = os.path.join(DATAPATH, dataset_path, 'processed', 
				f'{self.params.probing_task}.csv') # should also use config_filename?
			self.encoded_datapath = os.path.join(DATAPATH, dataset_path, 'processed', 
				f'{self.params.probing_task}') # should also use config_filename?
			self.result_datapath = os.path.join(LOGS_PATH, f'result_{config_filename}.csv')
			self.result_pred_datapath = os.path.join(LOGS_PATH, f'result_pred_{config_filename}.csv')
			self.fig_datapath = os.path.join(LOGS_PATH, f'fig_{config_filename}.png')

		if self.params.probing_task == 'mask':
			self.processed_datapath = os.path.join(DATAPATH, SEMEVAL_PATH, 'processed', 
				f'{self.params.probing_task}_{self.params.mask}.txt') # 'data/causal_probing/SemEval_2010_8/processed/mask_cause.txt'
			# self.processed_datapath = SemEval_mask_PROCESSED_DATAPATH + f'_{self.params.mask}_processed.txt'
			
			# e.g., TRIAL_mask_semeval_cause_555
			config_filename = ('TRIAL_' if self.params.trial else '') + \
				'_'.join([self.params.probing_task, self.params.dataset, self.params.mask, str(self.params.seed)])
			self.pred_datapath = os.path.join(LOGS_PATH, 'pred_' + config_filename) # without extension
			self.acc_datapath = os.path.join(LOGS_PATH, 'acc_' + config_filename)
			self.fig_datapath = os.path.join(LOGS_PATH, 'fig_' + config_filename)

		elif self.params.probing_task == 'feature':
			config_filename = ('TRIAL_' if self.params.trial else '') + \
				'_'.join([self.params.probing_task, self.params.dataset, str(self.params.seed)])
			if self.params.dataset == 'semeval':
				dataset_path = SEMEVAL_PATH
			# elif self.params.dataset == 'because':
			# 	dataset_path = BECAUSE_PATH
			elif self.params.dataset == 'roc':
				dataset_path = ROC_PATH

			self.processed_datapath = os.path.join(DATAPATH, dataset_path, 'processed', 
				f'{self.params.probing_task}.csv')  # should also use config_filename?
			self.encoded_datapath = os.path.join(DATAPATH, dataset_path, 'processed', 
				f'{self.params.probing_task}') # should also use config_filename?

			# self.result_datapath = LOGS_PATH + f'{self.params.dataset}_feature_result.csv'
			self.result_datapath = os.path.join(LOGS_PATH, f'result_{config_filename}.csv')
			self.fig_datapath = os.path.join(LOGS_PATH, f'fig_{config_filename}')

		self.last_filename = '{}_{}_{}_{}_{}'.format(
			'_TRIAL' if self.params.trial else '',
			self.params.dataset, self.params.probing_task, 
			self.params.mask, self.params.seed)

	def eval(self):
		if self.params.reset_data: # TEMP
			self.preprocess_data()

		self.load_data()
		self.prepare_data()

		if self.params.probing_task == 'simple':
			self.result = []
			self.result_pred = pd.DataFrame(columns=['X_idx', 'pred', 'true', 'model'])
			for model in ['bert', 'glove', 'gpt2', 'conceptnet']:
				encoded_model_path = self.encoded_datapath + f'_{model}.pkl'
				if self.params.reencode_data:
					self.encode(model, encoded_model_path)
				embeddings = utils.load_newest(encoded_model_path)
				logging.info(f'all encoded data loaded')

				logging.info(f'start training...')
				result_raw, result_pred = self.train(embeddings, self.data.causal, return_pred=True)
				for r in result_raw:
					r['model'] = model
				self.result += result_raw
				result_pred['model'] = [model] * len(result_pred['true'])
				self.result_pred = self.result_pred.append(pd.DataFrame(result_pred), ignore_index=True)
			self.result = pd.DataFrame(self.result, columns=['model', 'metric', 'split', 'value'])
			utils.save_dt(self.result, self.result_datapath)
			utils.save_dt(self.result_pred, self.result_pred_datapath)

			self.plot_metrics(self.fig_datapath)

		elif self.params.probing_task == 'mask':
			self.predict_mask()
			self.save_pred_mask()
			self.plot_acc_mask()

		elif self.params.probing_task == 'feature':

			if self.params.pretrained.model == 'ALL':
				iter_models = ['bert', 'gpt2', 'glove', 'conceptnet']
			else:
				iter_models = [self.params.pretrained.model]

			for model in iter_models:
				if self.params.reencode_data:
					self.encode(model, self.encoded_datapath + f'_{model}.pkl')
				self.embeddings = utils.load_newest(self.encoded_datapath + f'_{model}.pkl')

				logging.info(f'start predicting by {model}...')
				self.predict_feature(cv=self.params.cv)

				self.result['model'] = model
				try:
					self.result_models = self.result_models.append(self.result, ignore_index=True)
				except:
					self.result_models = self.result

			utils.save_dt(self.result_models, self.result_datapath)

			if self.params.use_pytorch:
				self.plot_feature_acc_by_rel(self.result_datapath, self.fig_datapath)
			else:
				self.plot_acc_f1(self.result_datapath)

		# elif self.params.probing_task == 'choice':
		# 	print('123')

	def preprocess_data(self):
		logging.info('preprocessing data...')
		if self.params.probing_task == 'simple':
			if self.params.dataset == 'semeval':
				dl = SemEval_feature_preprocess.DataLoader() # change to use feature_preprocess cuz its compatible
				dl.read(self.raw_datapath)
				dl.preprocess(trial=self.params.trial)

			elif self.params.dataset == 'because':
				dl = BECAUSE_preprocess.DataLoader()
				dl.read(self.raw_datapath)
				dl.preprocess()

			elif self.params.dataset == 'roc':
				dl = ROC_preprocess.DataLoader()
				dl.read(self.raw_datapath)
				dl.preprocess(trial=self.params.trial)
			dl.save_output(self.processed_datapath)

		elif self.params.probing_task == 'mask':
			if self.params.dataset == 'semeval':
				dl = SemEval_preprocess.DataLoader()
				dl.read(self.raw_datapath)
				dl.preprocess(self.params.probing_task, mask=self.params.mask)
				# dl.split(dev_prop=0.2, test_prop=0.2, seed=self.params.seed)
				dl.write(self.processed_datapath)

		elif self.params.probing_task == 'feature':
			if self.params.dataset == 'semeval':
				dl = SemEval_feature_preprocess.DataLoader()
				dl.read(self.raw_datapath)
			elif self.params.dataset == 'roc':
				dl = ROC_preprocess.DataLoader()
				dl.read(ROC_RAW_DATAPATH)

			dl.preprocess(trial=self.params.trial)
			if self.params.label_data == 'semeval':
				dl.calc_prob()
			elif self.params.label_data == 'oanc':
				dl.calc_prob_oanc(OANC_DATAPATH, trial=self.params.trial)

			dl.make_categorical(self.params.num_classes, self.params.num_classes_by)
			dl.save_output(self.processed_datapath)

	def load_data(self):
		if self.params.probing_task == 'mask':
			self.data = {'X_orig': [], 'y': [], 'rel': []}
			with io.open(self.processed_datapath, 'r', encoding='utf-8') as f:
				for line in f:
					# line = 'te	gardens	[CLS] The winery includes [MASK]. [SEP]	Component-Whole'
					line = line.rstrip().split('\t')
					self.data['y'].append(line[1])
					self.data['X_orig'].append(line[2])
					self.data['rel'].append(line[3])
			logging.info(f'Loaded {len(self.data["X_orig"])}')

		elif self.params.probing_task == 'simple' or self.params.probing_task == 'feature':
			self.data = utils.load_newest(self.processed_datapath)
			logging.info(f'Loaded {self.data.shape}')

	def prepare_data(self):
		if self.params.probing_task == 'simple':
			return

		elif self.params.probing_task == 'mask':
			self.data['X_shuf'], self.data['X_trunc'], self.data['X_shuf_trunc'] = [], [], []
			def tokenize_with_mask(x):
				x = re.sub(r'([.,!?;])', r' \1', x) # replace sth like '.' to ' .'
				x = x.split()

				# join like [MASK] [MASK]
				mask_index = [i for i in range(len(x)) if '[MASK]' in x[i]]
				x = x[:mask_index[0]] + [' '.join(x[mask_index[0]:mask_index[-1]+1])] + x[mask_index[-1]+1:]
				return x
			def shuffle(x):
				'''shuffle everything except [CLS], last character (period), [SEP]'''
				x = tokenize_with_mask(x)
				start_token, end_token, period = x.pop(0), x.pop(), x.pop()
				np.random.shuffle(x)
				x = [start_token] + x + [period, end_token]
				return ' '.join(x)
			def truncate(x):
				'''truncate the sentence to A B [MASK] C D, or A B C D [MASK], etc'''
				x = tokenize_with_mask(x)
				start_token, end_token = x.pop(0), x.pop()
				mask_index = [i for i in range(len(x)) if '[MASK]' in x[i]]
				# assert mask_index[-1] - mask_index[0] == len(mask_index) - 1, 'masks should be neighbor'
				
				keep = 2
				start_idx, end_idx = mask_index[0]-keep, mask_index[-1]+keep
				if start_idx < 0:
					end_idx += -start_idx
					start_idx = 0
				elif end_idx >= len(x):
					# print(end_idx, len(x))
					start_idx -= end_idx - len(x) + 1
					start_idx = 0 if start_idx < 0 else start_idx
				x = x[start_idx:end_idx+1]
				x = [start_token] + x + [end_token]
				# ignore this restriction when the sentence is short, or when mask is not in neighbor (after shuffle)
				if len(x) > 6 and mask_index[-1] - mask_index[0] == len(mask_index) - 1: 
					assert len(x) == 2 + len(mask_index) + keep*2
				return ' '.join(x)

			np.random.seed(self.params.seed)
			for i in range(len(self.data['X_orig'])):

				shuf = shuffle(self.data['X_orig'][i])
				self.data['X_shuf'].append(shuf)

				trunc = truncate(self.data['X_orig'][i])
				self.data['X_trunc'].append(trunc)

				shuf_trunc = truncate(shuf)
				self.data['X_shuf_trunc'].append(shuf_trunc)
			
			# print(self.data['X_orig'][7555])
			# print(self.data['X_shuf'][7555])
			# print(self.data['X_trunc'][7555])
			# print(self.data['X_shuf_trunc'][7555])

		elif self.params.probing_task == 'feature':
			# # self.backup_data = self.data.copy()
			# # self.data = self.data.drop(columns=['cause', 'effect', 'c_count', 'e_count', 'c_e_count', 'e_no_c_count'])
			# use_cols = ['X', 'relation'] + [x if x in self.binary_columns else x+'_cat' for x in self.all_target_columns]
			# use_cols = [c for c in self.data.columns if c in use_cols] # this excludes -inf ['ovr_freq_tri_cat', 'avg_freq_tri_cat', 'ovr_freq_uni_cat', 'ovr_freq_bi_cat']
			# self.data = self.data[use_cols]
			# # self.data.columns = ['X', 'relation'] + self.all_target_columns
			# self.data.columns = [c[:-4] if '_cat' in c else c for c in self.data.columns]
			# # self.all_target_columns = self.data.columns[:3]
			# # print(self.data.columns)

			use_cols = ['X', 'relation'] + [c for c in self.data.columns if '_cat' in c]
			self.data = self.data[use_cols]

			self.all_target_columns = [c[:-4] for c in self.data.columns if '_cat' in c]
			self.data.columns = [c if '_cat' not in c else c[:-4] for c in self.data.columns]

	def encode(self, model, save_path):
		'''prepare different encoders and save to save_path'''
		logging.info(f'preparing encoder {model}...')
		params = self.params

		if model == 'bert':
			bert_type = f'bert-{params.pretrained.model_type}-{"cased" if params.pretrained.cased else "uncased"}'
			logging.getLogger('transformers').setLevel(logging.ERROR)
			params.tokenizer = BertTokenizer.from_pretrained(bert_type)
			# if self.params.probing_task == 'mask':
			# 	params.encoder = BertForMaskedLM.from_pretrained(bert_type).to(DEVICE)
			# else:
			# 	params.encoder = BertModel.from_pretrained('bert-base-uncased')
			params.encoder = BertModel.from_pretrained(bert_type)
			params.encoder.eval() # ??

		elif model == 'gpt2':
			logging.getLogger('transformers').setLevel(logging.ERROR)
			params.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
			params.encoder = GPT2Model.from_pretrained('gpt2')

		elif model == 'glove':
			samples = self.data['X']
			samples = [s.split() for s in samples]
			# Create dictionary
			def create_dictionary(sentences, threshold=0):
				words = {}
				for s in sentences:
					for word in s:
						words[word] = words.get(word, 0) + 1

				if threshold > 0:
					newwords = {}
					for word in words:
						if words[word] >= threshold:
							newwords[word] = words[word]
					words = newwords
				words['<s>'] = 1e9 + 4
				words['</s>'] = 1e9 + 3
				words['<p>'] = 1e9 + 2

				sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
				id2word = []
				word2id = {}
				for i, (w, _) in enumerate(sorted_words):
					id2word.append(w)
					word2id[w] = i

				return id2word, word2id
			_, params.word2id = create_dictionary(samples)
			# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
			def get_wordvec(path_to_vec, word2id):
				word_vec = {}

				with io.open(path_to_vec, 'r', encoding='utf-8') as f:
					# if word2vec or fasttext file : skip first line "next(f)"
					for line in f:
						word, vec = line.split(' ', 1)
						if word in word2id:
							word_vec[word] = np.fromstring(vec, sep=' ')

				logging.info('Found {0} words with word vectors, out of {1} words'.format(len(word_vec), len(word2id)))
				return word_vec
			params.word_vec = get_wordvec(GLOVE_PATH, params.word2id)
			params.wvec_dim = 300
			logging.debug(f'word2id: {params.word2id["man"]}') # 90
			logging.debug(f'word2vec: {params.word_vec["man"][:5]}') # 300

		elif model == 'conceptnet':
			import gzip

			with gzip.open(CONCEPTNET_PATH, 'rb') as f:
				lines = f.read()
			lines = lines.decode("utf-8").split('\n')[1:]

			params.word_vec = {}
			for l in lines:
				s = l.split()
				if s:
					params.word_vec[s[0]] = [float(x) for x in s[1:]]

			logging.debug(f'word2vec: {params.word_vec["man"][:5]}') # 300

		logging.info(f'prepared encoder {model}')


		logging.info(f'encoding data by {model}...')
		if model == 'bert' or model == 'gpt2':
			hidden_size = self.params.encoder.config.hidden_size
			embeddings = torch.zeros(self.data.shape[0], hidden_size)
			pb = utils.ProgressBar(len(self.data.X))
			for i in range(len(self.data.X)):
				pb.now(i)

				if model == 'bert':
					x = '[CLS] ' + self.data.X[i] + ' [SEP]'
				elif model == 'gpt2':
					x = self.data.X[i]

				tokenized_text = self.params.tokenizer.tokenize(x)
				# double check tokenize correctly, previous version of transformers has some problems
				if self.data.X[i] == 'The system as described above has its greatest application in an arrayed configuration of antenna elements.':
					if model == 'bert':
						assert tokenized_text == ['[CLS]', 'the', 'system', 'as', 'described', 'above', 'has', 'its', 'greatest', 'application', 'in', 'an', 'array', '##ed', 'configuration', 'of', 'antenna', 'elements', '.', '[SEP]']
					elif model == 'gpt2':
						assert tokenized_text == ['The', 'Ġsystem', 'Ġas', 'Ġdescribed', 'Ġabove', 'Ġhas', 'Ġits', 'Ġgreatest', 'Ġapplication', 'Ġin', 'Ġan', 'Ġarray', 'ed', 'Ġconfiguration', 'Ġof', 'Ġantenna', 'Ġelements', '.']

				indexed_tokens = self.params.tokenizer.convert_tokens_to_ids(tokenized_text)
				tokens_tensor = torch.tensor([indexed_tokens])

				with torch.no_grad():
					outputs = self.params.encoder(tokens_tensor)
					encoded_layers = outputs[0]

				embeddings[i, :] = torch.mean(encoded_layers, dim=1)
				# print(encoded_layers.shape)

				if self.params.trial:
					break
			embeddings = embeddings.numpy()

		elif model == 'glove' or model == 'conceptnet':
			embeddings = []
			for sent in self.data.X:
				# tokenize
				sent = [w.lower() for w in word_tokenize(sent)]
				# print(sent) # ['the', 'system', 'as', 'described', 'above', 'has', 'its', 'greatest', 'application', 'in', 'an', 'arrayed', 'configuration', 'of', 'antenna', 'elements', '.']

				sentvec = []
				for word in sent:
					if word in params.word_vec:
						sentvec.append(params.word_vec[word])
				if not sentvec:
					vec = np.zeros(params.wvec_dim)
					sentvec.append(vec)
				sentvec = np.mean(sentvec, 0)
				embeddings.append(sentvec)

				if self.params.trial:
					break

			embeddings = np.vstack(embeddings)

		utils.save_dt(embeddings, save_path)
		logging.info(f'data encoded by {model}, embeddings shape: {embeddings.shape}')
		# return embeddings

	def predict_mask(self):
		logging.info('predicting...')
		k_list = [1, 3, 5, 7, 9, 10, 20]
		X_types = ['X_orig', 'X_shuf', 'X_trunc', 'X_shuf_trunc']
		correct = {rel:{k:{X_type:[] for X_type in X_types} for k in k_list} for rel in list(set(self.data['rel']))}
		self.pred = []
		
		pb = utils.ProgressBar(len(self.data['X_orig']))
		for i in range(len(self.data['X_orig'])):
			pb.now(i)
			pred = []
			y = self.data['y'][i]
			rel = self.data['rel'][i]
			for X_type in X_types:
				x = self.data[X_type][i]

				# print(x)
				tokenized_text = self.params.tokenizer.tokenize(x)
				# print(tokenized_text)
				indexed_tokens = self.params.tokenizer.convert_tokens_to_ids(tokenized_text)
				# print(indexed_tokens)
				masked_index = [i for i in range(len(tokenized_text)) if tokenized_text[i] == '[MASK]']
				# print(masked_index)
				tokens_tensor = torch.tensor([indexed_tokens])
				tokens_tensor = tokens_tensor.to(DEVICE)
				# print(tokens_tensor.shape)

				with torch.no_grad():
					outputs = self.params.encoder(tokens_tensor)
					# print(len(outputs))
					predictions = outputs[0]
				# print(predictions.shape) # 1, 19, 768
				# print(predictions[0, masked_index].shape) # 2, 768
				soft_pred = torch.softmax(predictions[0, masked_index], 1)
				# print(soft_pred.shape) # 2, 768
				top_inds = torch.argsort(soft_pred, descending=True)[:, :k_list[-1]].cpu().numpy()
				# print(top_inds.shape) # 2, 5
				# top_probs = [soft_pred[tgt_ind].item() for tgt_ind in top_inds]
				top_k_preds = [self.params.tokenizer.convert_ids_to_tokens(top_inds[to_pred, :]) \
					for to_pred in range(top_inds.shape[0])]
				# print(top_k_preds)
				# print(y)
				# print()

				correct_at = []
				for j, y_j in enumerate(y.split()):
					# print(top_k_preds[j], y_j)
					temp = [l+1 for l in range(k_list[-1]) if top_k_preds[j][l] == y_j]
					# print('temp', temp)
					correct_at.append(temp[0] if temp != [] else 0)
				# print(correct_at)
				correct_at = max(correct_at) if min(correct_at) != 0 else 0
				# print('correct_at', correct_at)

				for k in k_list:
					# num_correct = [y.split()[j] in top_k_preds[j][:k] for j in range(len(masked_index))]
					# print(int(correct_at <= k and correct_at != 0))
					correct[rel][k][X_type].append(int(correct_at <= k and correct_at != 0))

				pred += [x, top_k_preds, correct_at]
			pred = [y, rel] + pred
			self.pred.append(pred)

			if self.params.trial:
				# print(pred)
				# print()
				if i == 8:
					break
		self.acc = {k1:{k2:{k3:np.mean(v3) if v3 != [] else 0 for k3, v3 in v2.items()} for k2, v2 in v1.items()} for k1, v1 in correct.items()}
		# logging.info(f'acc: {self.acc["Cause-Effect"][5]}')

	def predict_feature(self, cv):
		from sklearn.model_selection import cross_validate
		from sklearn.linear_model import LogisticRegression

		# from sklearn.utils.testing import ignore_warnings
		from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
		import warnings
		warnings.filterwarnings("ignore", category=FutureWarning)
		warnings.filterwarnings("ignore", category=ConvergenceWarning)
		warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

		result = []
		for rel in self.data.relation.unique():
			if rel != 'Cause-Effect': # TEMP only predict cause-effect so tons faster
				continue
			for y_name in self.all_target_columns:
				X = self.embeddings[np.array(self.data.relation == rel), :]
				y = self.data.loc[self.data.relation == rel, y_name]
				if y.nunique() == 1:
					print(rel, y_name, 'y are all', y.iloc[0])
					continue
				if any(y.value_counts() <= 5):
					print(f'{rel}, {y_name} y doesnt have enough sample in one of the classes')
					print(y.value_counts())
					print()
					continue

				if self.params.use_pytorch:
					logging.info(f'start training {rel} {y_name}')
					result_raw = self.train(X, y)
					for r in result_raw:
						r.update({'relation': rel, 'y_type': y_name})
					# print(result_raw)
					result += result_raw

				else: # original, should be merge into train()
					# check nan
					if y.isnull().sum() > 0:
						logging.info(f'    removing {y.isnull().sum()} nulls in y')
						X = X[y.notnull()]
						y = y[y.notnull()]

					y = np.array(y)

					clf = LogisticRegression(solver='lbfgs', random_state=self.params.seed) # if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.

					if len(np.unique(y)) == 2:
						scoring = ('accuracy', 'balanced_accuracy', 'f1', 
							'precision', 'recall', 'roc_auc')
					else:
						scoring = ('accuracy', 'balanced_accuracy', 'f1_weighted', 
							# 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted'
							)
					cv_results = cross_validate(clf, X, y, cv=cv, error_score='raise',
						scoring=scoring) # error_score=np.nan
					clf.fit(X, y)

					# print(cv_results)
					result_temp = {k:cv_results[f'test_{k}'].mean() for k in scoring}
					result_temp.update({'relation': rel, 'y_type': y_name})
					result.append(result_temp)
				# break
			# break
		self.result = pd.DataFrame(result)
		# to_move = ['relation', 'y_type']
		# result = result[['relation', 'y_type'] + result.columns.tolist().remove(['relation', 'y_type'])]
		
		# utils.save_dt(result, index=False)

	def train(self, embeddings, y, return_pred=False):
		from sklearn.model_selection import cross_validate
		from sklearn.linear_model import LogisticRegression

		from sklearn.model_selection import train_test_split
		from sklearn.model_selection import StratifiedKFold

		# check nan
		if y.isnull().sum() > 0:
			logging.info(f'    removing {y.isnull().sum()} nulls in y')
			embeddings = embeddings[y.notnull()]
			y = y[y.notnull()]

		X = embeddings
		y = np.array(y)
		assert X.shape[0] == len(y), f'X shape {X.shape} and y len {len(y)} not compatible'

		# logging.info('start training...')
		if self.params.use_pytorch:
			# TODO default settings, should be add to argparse too
			# self.classifier_config = config['classifier']
			self.classifier_config = {'nhid': 0} # will use default settings in MLP?, nhid = 0 means logistic regression
			self.featdim = X.shape[1]
			self.nclasses = len(np.unique(y))
			# reg = 1e-9 # no reg, or for reg in [10**t for t in range(-5, -1)]
			self.cudaEfficient = False # if 'cudaEfficient' not in config else config['cudaEfficient']

			# train dev test split k-fold
			result_raw = []
			result_pred = {'X_idx': [], 'pred': [], 'true': []}
			skf = StratifiedKFold(n_splits=self.params.cv, shuffle=True, random_state=self.params.seed)
			skf.get_n_splits(X, y)
			for train_index, test_index in skf.split(X, y):
				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = y[train_index], y[test_index]
				X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=self.params.seed)
				# print(X_train.shape, X_val.shape, X_test.shape)

				# training
				regs = [10**t for t in range(-5, -1)]
				scores = []
				for reg in regs:
					clf = MLP(self.classifier_config, inputdim=self.featdim,
						nclasses=self.nclasses, l2reg=reg, # reg will be passed to weight_decay in torch.optim.Adam(), lead to to smaller model weights
						seed=self.params.seed, cudaEfficient=self.cudaEfficient)
					clf.fit(X_train, y_train, validation_data=(X_val, y_val))
					scores.append(clf.score(X_val, y_val))
				# logging.info([('reg:' + str(regs[idx]), scores[idx]) for idx in range(len(scores))])
				optreg = regs[np.argmax(scores)]
				devaccuracy = np.max(scores)
				# logging.info('Validation : best param found is reg = {0} with score {1}'.format(optreg, devaccuracy))

				# re train
				clf = MLP(self.classifier_config, inputdim=self.featdim,
					nclasses=self.nclasses, l2reg=optreg,
					seed=self.params.seed, cudaEfficient=self.cudaEfficient)
				clf.fit(X_train, y_train, validation_data=(X_val, y_val))

				# test
				testaccuracy, pred = clf.score(X_test, y_test, return_pred=True)

				# record all prediction for analysis
				result_pred['X_idx'] += test_index.tolist()
				result_pred['pred'] += pred
				result_pred['true'] += y_test.tolist()
				assert len(result_pred['pred']) == len(result_pred['true']), 'pred and true y have diff len, pred: {}, true: {}'.format(len(result_pred['pred']), len(result_pred['true']))

				result_raw += [
					{'metric': 'accuracy', 'split': 'dev', 'value': devaccuracy},
					{'metric': 'accuracy', 'split': 'test', 'value': testaccuracy}]
			
			# logging.info('done testing')
			if not return_pred:
				return result_raw
			else:
				return result_raw, result_pred

		else: # use sklearn
			# SentEval use
			# LogisticRegression(C=reg, random_state=self.seed) 
			# reg = [2**t for t in range(-2, 4, 1)], 2^-2 ~ 2^4, 0.25 ~ 16, default of LogisticRegressionCV is 1e-4 ~ 1e4 (10 steps)
			clf = LogisticRegression(solver='lbfgs', random_state=self.params.seed,
				# class_weight='balanced', # TODO not sure
				max_iter=1000) # if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
			
			if len(np.unique(y)) == 2: # binary
				scoring = ('accuracy', 'balanced_accuracy', 'f1', 
					'precision', 'recall', 'roc_auc')
			else: # multiclass
				scoring = ('accuracy', 'balanced_accuracy', 'f1_weighted')
			cv_results = cross_validate(clf, X, y, cv=self.params.cv, error_score='raise', scoring=scoring,
				n_jobs=5) # error_score=np.nan

			result_raw = [{'metric': k, 'value': v} for k in scoring for v in cv_results[f'test_{k}']]
			logging.info('done training')
			return result_raw

	def save_pred_mask(self):
		lines_pred = ['\t'.join([str(x) for x in l]) + '\n' for l in self.pred]
		utils.save_dt(lines_pred, self.pred_datapath + '.txt')
		utils.save_dt(self.pred, self.pred_datapath + '.pkl')

		lines_acc = [k + ' : ' + str(v) + '\n' for k, v in self.acc.items()]
		utils.save_dt(lines_acc, self.acc_datapath + '.txt')
		utils.save_dt(self.acc, self.acc_datapath + '.pkl')

	def plot_acc_mask(self):
		acc = utils.load_newest(self.acc_datapath + '.pkl')
		
		# process to sns form
		d = pd.DataFrame()
		for rel in acc:
			for k in acc[rel]:
				for X_type in acc[rel][k]:
					d = d.append({'rel': rel, 'k': k, 'X_type': X_type, 
						'acc': acc[rel][k][X_type]}, ignore_index=True)

		for rel in d.rel.unique():
			g = sns.catplot(x='k', y='acc', hue='X_type', 
				hue_order=['X_orig', 'X_trunc', 'X_shuf', 'X_shuf_trunc'],
				data=d[d.rel == rel], kind='bar',
				palette='Set1')
			# g.savefig(path + 'fig_' + rel + self.last_filename + '.png',
			# 	dpi=600, bbox_inches='tight')
			utils.save_dt(g, self.fig_datapath + f'_{rel}.png', dpi=600, bbox_inches='tight')

		# logging.info('fig saved')

	def plot_acc_f1(self, result_path):
		d = utils.load_newest(result_path)

		# merge f1 and f1_weighted to f1
		if 'f1' in d.columns:
			d['f1_binary'] = d.f1
			d.loc[d.f1.isnull(), 'f1'] = d.loc[d.f1.isnull(), 'f1_weighted']
		else:
			d['f1'] = d['f1_weighted']

		def plot_metric(d, metric, only_causal):
			if not only_causal:
				g = sns.catplot(y=metric, x='y_type', hue='model', col='relation', col_wrap=3, data=d, kind='bar')
			else:
				g = sns.catplot(y=metric, x='y_type', hue='model', data=d[d.relation == 'Cause-Effect'], kind='bar')
			for ax in g.axes.flat: 
				for label in ax.get_xticklabels():
					label.set_rotation(45)
					label.set_ha('right')
			filename = LOGS_PATH + f'fig_{self.params.probing_task}_{self.params.dataset}_{metric}{"_causal" if only_causal else ""}_{self.params.seed}.png'
			utils.save_dt(plt, filename, dpi=600, bbox_inches='tight')
		plot_metric(d, 'f1', False)
		plot_metric(d, 'balanced_accuracy', False)
		plot_metric(d, 'f1', True)
		plot_metric(d, 'balanced_accuracy', True)

	def plot_feature_acc_by_rel(self, result_path, fig_path):
		def rotate_xlabels(g):
			for ax in g.axes.flat: 
				for label in ax.get_xticklabels():
					label.set_rotation(45)
					label.set_ha('right')
		def show_values_on_bars(g, height_adjust=0):
			for ax in g.axes.flat: 
				max_y = max([p.get_y() + p.get_height() for p in ax.patches])
				for p in ax.patches:
					_x = p.get_x() + p.get_width() / 2
					_y = p.get_y() + p.get_height() + max_y * height_adjust/100
					value = '{:.2f}'.format(p.get_height())
					ax.text(_x, _y, value, ha="center") 
		self.result = utils.load_newest(result_path)
		self.result = self.result[self.result.y_type != 'causal_dependency']  # TEMP setting # remove causal_dependency because imbalanced
		self.result = self.result[self.result.split == 'test'] # TEMP plot only for test set
		for rel in self.result.relation.unique():
			g = sns.catplot(y='value', x='y_type', hue='model', 
				data=self.result[self.result.relation == rel], 
				kind='bar', col='split')
			rotate_xlabels(g)
			show_values_on_bars(g, 3)
			utils.save_dt(g, fig_path + f'_{rel}.png')

	def plot_metrics(self, fig_datapath):
		if self.params.use_pytorch:
			g = sns.catplot(y='value', x='split', hue='model', data=self.result, kind='bar')
		else:
			g = sns.catplot(y='value', x='metric', hue='model', data=self.result, kind='bar')
		def rotate_xlabels(g):
			for ax in g.axes.flat: 
				for label in ax.get_xticklabels():
					label.set_rotation(45)
					label.set_ha('right')
		def show_values_on_bars(g, height_adjust=0):
			for ax in g.axes.flat: 
				max_y = max([p.get_y() + p.get_height() for p in ax.patches])
				for p in ax.patches:
					_x = p.get_x() + p.get_width() / 2
					_y = p.get_y() + p.get_height() + max_y * height_adjust/100
					value = '{:.2f}'.format(p.get_height())
					ax.text(_x, _y, value, ha="center") 
		rotate_xlabels(g)
		show_values_on_bars(g, 3)
		utils.save_dt(g, fig_datapath)