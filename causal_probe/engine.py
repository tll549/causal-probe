import io
import numpy as np
import pandas as pd
import logging
import os
import re
import pickle

from causal_probe import utils
from data import SemEval_preprocess
from data import feature_preprocess # also for SemEval
from data import ROC_preprocess

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from transformers import BertTokenizer, BertModel, BertForMaskedLM

import matplotlib.pyplot as plt
import seaborn as sns

PATH_TO_VEC = 'examples/glove/glove.840B.300d.txt'

SemEval_RAW_DATAPATH = 'data/causal_probing/SemEval_2010_8/raw/TRAIN_FILE.TXT'
SemEval_mask_PROCESSED_DATAPATH = 'data/causal_probing/SemEval_2010_8/processed/SemEval_mask'
SemEval_LOGS_DATAPATH = 'logs/'

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

		if self.params.probing_task == 'mask':
			self.processed_datapath = SemEval_mask_PROCESSED_DATAPATH + f'_{self.params.mask}_processed.txt'
		elif self.params.probing_task == 'feature':
			self.result_datapath = SemEval_LOGS_DATAPATH + f'{self.params.dataset}_feature_result.csv'
			if self.params.dataset == 'semeval':
				self.processed_datapath = SemEval_feature_PROCESSED_DATAPATH
				self.encoded_datapath = SemEval_feature_ENCODED_DATAPATH
			elif self.params.dataset == 'roc':
				self.processed_datapath = ROC_feature_PROCESSED_DATAPATH
				self.encoded_datapath = ROC_feature_ENCODED_DATAPATH

			self.all_target_columns = ['causal_dependency', 'P(E|C)', 'P(E)', 
				# 'probabilistic_causality', 
				'probabilistic_causality_diff', 
				'delta_P', 'P(E|no C)', 'q', 'p', 'causal_power']
			self.numerical_columns = ['P(E|C)', 'P(E)', 'probabilistic_causality_diff',
				'delta_P', 'P(E|no C)', 'q', 'p', 'causal_power']
			self.binary_columns = [x for x in self.all_target_columns if x not in self.numerical_columns]
		# elif self.params.probing_task == 'choice':
		# 	pass

		self.last_filename = '{}_{}_{}_{}_{}'.format(
			'_TRIAL' if self.params.trial else '',
			self.params.dataset, self.params.probing_task, 
			self.params.mask, self.params.seed)

	def eval(self):
		if self.params.reset_data:
			self.preprocess_data()

		self.load_data()
		self.prepare_data()
		self.prepare_encoder()

		if self.params.probing_task == 'simple':
			# not done yet
			pass

		elif self.params.probing_task == 'mask':
			self.predict_mask()
			self.save_pred(SemEval_LOGS_DATAPATH)
			self.plot_acc(SemEval_LOGS_DATAPATH)

		elif self.params.probing_task == 'feature':

			if self.params.pretrained.model == 'bert': # or both
				self.encode_data_bert(self.encoded_datapath)
				# if self.params.reset_data:
				# 	self.encode_data_bert(self.encoded_datapath)
				# else:
				# 	self.load_encoded_data(self.encoded_datapath)
				logging.info('start predicting by bert...')
				self.predict_feature(cv=self.params.cv)

				self.result['model'] = 'bert'
				self.result_bert_glove = self.result.copy()

			if self.params.pretrained.model == 'glove' or self.params.pretrained.both_bert_glove:
				# if self.params.reset_data:
				self.encode_data_glove(self.encoded_datapath)
				# else:
				# 	self.load_encoded_data(self.encoded_datapath)

				logging.info('start predicting by glove...')
				self.predict_feature(cv=self.params.cv)

				self.result['model'] = 'glove'
				try:
					self.result_bert_glove = self.result_bert_glove.append(self.result, ignore_index=True)
				except:
					self.result_bert_glove = self.result

			utils.save_dt(self.result_bert_glove, self.result_datapath)

			self.plot_acc_f1(self.result_datapath)

		# elif self.params.probing_task == 'choice':
		# 	print('123')

	def preprocess_data(self):
		logging.info('preprocessing data...')
		if self.params.probing_task == 'mask':
			if self.params.dataset == 'semeval':
				dl = SemEval_preprocess.DataLoader()
				dl.read(SemEval_RAW_DATAPATH)
				dl.preprocess(self.params.probing_task, mask=self.params.mask)
				# dl.split(dev_prop=0.2, test_prop=0.2, seed=self.params.seed)
				dl.write(self.processed_datapath)
		elif self.params.probing_task == 'feature':
			if self.params.dataset == 'semeval':
				dl = feature_preprocess.DataLoader()
				dl.read(SemEval_RAW_DATAPATH)
			elif self.params.dataset == 'roc':
				dl = ROC_preprocess.DataLoader()
				dl.read(ROC_RAW_DATAPATH)

			dl.preprocess(trial=self.params.trial)
			if self.params.label_data == 'semeval':
				dl.calc_prob()
			elif self.params.label_data == 'oanc':
				dl.calc_prob_oanc(OANC_DATAPATH)

			dl.make_categorical(self.params.num_classes, self.params.num_classes_by,
				self.numerical_columns)
			dl.save_output(self.processed_datapath)

	def load_data(self):
		# if self.params.dataset == 'semeval':
		# 	fpath = 'data/causal_probing/SemEval_2010_8/processed/SemEval_mask_processed.txt'
		# elif self.params.dataset == 'because':
		# 	fpath = 'data/causal_probing/BECAUSE/processed/because_all.txt'

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
		elif self.params.probing_task == 'feature':
			self.data = utils.load_newest(self.processed_datapath)
			logging.info(f'Loaded {self.data.shape}')

	def prepare_data(self):
		if self.params.probing_task == 'mask':
			self.data['X_shuf'], self.data['X_trunc'], self.data['X_shuf_trunc'] = [], [], []
			def tokenize_with_mask(x):
				# x = 'eroj ewriow eio. fwiej; fweji, wdej! erijf?'
				# print(x)
				x = re.sub(r'([.,!?;])', r' \1', x) # replace sth like '.' to ' .'
				# print(x)
				x = x.split()
				# print(x)
				# separate like [MASK]. to [MASK] .
				# mask_index = [i for i in range(len(x)) if '[MASK]' in x[i] and x[i] != '[MASK]']
				# for i in mask_index:
				# 	x.insert(i+1, re.sub(r'\[MASK\]', '', x[i]))
				# 	x[i] = '[MASK]'
				# period = x[-2][-1]
				# x[-2] = x[-2][:-1]
				# x.insert(-1, period)

				# print(x)
				# join like [MASK] [MASK]
				mask_index = [i for i in range(len(x)) if '[MASK]' in x[i]]
				x = x[:mask_index[0]] + [' '.join(x[mask_index[0]:mask_index[-1]+1])] + x[mask_index[-1]+1:]
				return x
			def shuffle(x):
				'''shuffle everything except [CLS], last character (period), [SEP]'''
				x = tokenize_with_mask(x)
				# print(x)
				start_token, end_token, period = x.pop(0), x.pop(), x.pop()
				# print(x)
				np.random.shuffle(x)
				# x[-1] = x[-1] + period
				x = [start_token] + x + [period, end_token]
				# print(x)
				# print(' '.join(x))
				# print()
				return ' '.join(x)
			def truncate(x):
				'''truncate the sentence to A B [MASK] C D, or A B C D [MASK], etc'''
				x = tokenize_with_mask(x)
				start_token, end_token = x.pop(0), x.pop()
				mask_index = [i for i in range(len(x)) if '[MASK]' in x[i]]
				# assert mask_index[-1] - mask_index[0] == len(mask_index) - 1, 'masks should be neighbor'
				
				keep = 2
				# print(mask_index)
				start_idx, end_idx = mask_index[0]-keep, mask_index[-1]+keep
				# print(start_idx, end_idx)
				if start_idx < 0:
					end_idx += -start_idx
					start_idx = 0
				elif end_idx >= len(x):
					# print(end_idx, len(x))
					start_idx -= end_idx - len(x) + 1
					start_idx = 0 if start_idx < 0 else start_idx
				# print(start_idx, end_idx)
				# print()
				x = x[start_idx:end_idx+1]
				# period = x.pop()
				# x[-1] += period
				x = [start_token] + x + [end_token]
				# print(x)
				# ignore this restriction when the sentence is short, or when mask is not in neighbor (after shuffle)
				if len(x) > 6 and mask_index[-1] - mask_index[0] == len(mask_index) - 1: 
					assert len(x) == 2 + len(mask_index) + keep*2
				return ' '.join(x)

			np.random.seed(self.params.seed)
			for i in range(len(self.data['X_orig'])):
				# print(self.data['X_orig'][i])

				shuf = shuffle(self.data['X_orig'][i])
				self.data['X_shuf'].append(shuf)
				# print(shuf)

				trunc = truncate(self.data['X_orig'][i])
				self.data['X_trunc'].append(trunc)
				# print(trunc)

				shuf_trunc = truncate(shuf)
				self.data['X_shuf_trunc'].append(shuf_trunc)
				# print(shuf_trunc)
				# print()
				# break
			
			# print(self.data['X_orig'][7555])
			# print(self.data['X_shuf'][7555])
			# print(self.data['X_trunc'][7555])
			# print(self.data['X_shuf_trunc'][7555])

		elif self.params.probing_task == 'feature':
			self.backup_data = self.data.copy()
			# self.data = self.data.drop(columns=['cause', 'effect', 'c_count', 'e_count', 'c_e_count', 'e_no_c_count'])
			use_cols = ['X', 'relation'] + [x if x in self.binary_columns else x+'_cat' for x in self.all_target_columns]
			self.data = self.data[use_cols]
			self.data.columns = ['X', 'relation'] + self.all_target_columns
			# print(self.data.head())

	def prepare_encoder(self):
		logging.info('preparing encoder...')
		params = self.params

		if params.pretrained.model == 'bert': # no need or params.pretrained.both_bert_glove
			bert_type = f'bert-{params.pretrained.model_type}-{"cased" if params.pretrained.cased else "uncased"}'
			logging.getLogger('transformers').setLevel(logging.ERROR)
			params.tokenizer = BertTokenizer.from_pretrained(bert_type)
			if self.params.probing_task == 'mask':
				params.encoder = BertForMaskedLM.from_pretrained(bert_type).to(DEVICE)
			elif self.params.probing_task == 'feature':
				params.encoder = BertModel.from_pretrained('bert-base-uncased')
			params.encoder.eval() # ??

		if params.pretrained.model == 'glove' or params.pretrained.both_bert_glove:
			# from senteval
			if 'train' in self.data:
				samples = self.data['train']['X'] + self.data['dev']['X'] + self.data['test']['X']
			else:
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
			params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
			params.wvec_dim = 300
			logging.debug(f'word2id: {params.word2id["man"]}') # 90
			logging.debug(f'word2vec: {params.word_vec["man"][:5]}') # 300

		logging.info('prepared')

	def encode_data_bert(self, save_path):
		logging.info('encoding data...')

		hidden_size = self.params.encoder.config.hidden_size
		self.embeddings = torch.zeros(self.data.shape[0], hidden_size)
		for i in range(len(self.data.X)):
			x = '[CLS] ' + self.data.X[i] + ' [SEP]'

			tokenized_text = self.params.tokenizer.tokenize(x)
			indexed_tokens = self.params.tokenizer.convert_tokens_to_ids(tokenized_text)
			tokens_tensor = torch.tensor([indexed_tokens])

			with torch.no_grad():
				outputs = self.params.encoder(tokens_tensor)
				encoded_layers = outputs[0]

			self.embeddings[i, :] = torch.mean(encoded_layers, dim=1)
			# print(encoded_layers.shape)

			if self.params.trial:
				break
		self.embeddings = self.embeddings.numpy()

		utils.save_dt(self.embeddings, save_path)
		logging.info(f'data encoded, embeddings shape: {self.embeddings.shape}')

	def encode_data_glove(self, save_path):
		params = self.params

		self.embeddings = []
		for sent in self.data.X:
			sentvec = []
			for word in sent:
				if word in params.word_vec:
					sentvec.append(params.word_vec[word])
			if not sentvec:
				vec = np.zeros(params.wvec_dim)
				sentvec.append(vec)
			sentvec = np.mean(sentvec, 0)
			self.embeddings.append(sentvec)

		self.embeddings = np.vstack(self.embeddings)
		# print(embeddings.shape) # (128, 300)
		# print(embeddings)
		return self.embeddings

	def load_encoded_data(self, save_path):
		self.embeddings = utils.load_newest(save_path)
		logging.info(f'loaded, embeddings shape {self.embeddings.shape}')

	def predict_mask(self):
		logging.info('predicting...')
		# k = self.params.k
		k_list = [1, 3, 5, 7, 9, 10, 20]
		X_types = ['X_orig', 'X_shuf', 'X_trunc', 'X_shuf_trunc']
		correct = {rel:{k:{X_type:[] for X_type in X_types} for k in k_list} for rel in list(set(self.data['rel']))}
		self.pred = []
		
		for i in range(len(self.data['X_orig'])):
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

				with torch.no_grad():
					outputs = self.params.encoder(tokens_tensor)
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
				print(pred)
				print()
				if i == 8:
					break
		self.acc = {k1:{k2:{k3:np.mean(v3) if v3 != [] else 0 for k3, v3 in v2.items()} for k2, v2 in v1.items()} for k1, v1 in correct.items()}
		# logging.info(f'acc: {self.acc["Cause-Effect"][5]}')

	def predict_feature(self, cv):
		from sklearn.model_selection import cross_validate
		from sklearn.linear_model import LogisticRegression

		from sklearn.utils.testing import ignore_warnings
		from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

		import warnings
		warnings.filterwarnings("ignore", category=FutureWarning)
		warnings.filterwarnings("ignore", category=ConvergenceWarning)
		warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

		result = []
		for rel in self.data.relation.unique():
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
				y = np.array(y)
				# print(rel, X.shape, y.shape, y_name)

				clf = LogisticRegression(solver='lbfgs', random_state=self.params.seed) # if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.

				if len(np.unique(y)) == 2:
					scoring = ('accuracy', 'balanced_accuracy', 'f1', 
						'precision', 'recall', 'roc_auc')
				else:
					scoring = ('accuracy', 'balanced_accuracy', 'f1_weighted', 
						# 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted'
						)
				# print(type(X), type(y))
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

	def save_pred(self, path):
		with open(path + 'result' + self.last_filename + '.txt', 'w+', encoding='utf-8') as f:
			for l in self.pred:
				o = '\t'.join([str(x) for x in l]) + '\n'
				f.write(o)
		with open(path + 'result' + self.last_filename + '.pkl', 'wb') as f:
			pickle.dump(self.pred, f)

		with open(path + 'acc' + self.last_filename + '.txt', 'w+', encoding='utf-8') as f:
			for k, v in self.acc.items():
				f.write(k + ' : ' + str(v) + '\n')
		with open(path + 'acc' + self.last_filename + '.pkl', 'wb') as f:
			pickle.dump(self.acc, f)

		logging.info('files saved')

	def plot_acc(self, path):
		with open(path + 'acc' + self.last_filename + '.pkl', 'rb') as f:
			acc = pickle.load(f)
		
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
			g.savefig(path + 'fig_' + rel + self.last_filename + '.png',
				dpi=600, bbox_inches='tight')

		logging.info('fig saved')

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
			filename = SemEval_LOGS_DATAPATH + f'fig_{self.params.dataset}_{self.params.probing_task}_{metric}{"_causal" if only_causal else ""}_{self.params.seed}.png'
			# plt.savefig(filename, dpi=600, bbox_inches='tight')
			# plt.show()
			utils.save_dt(plt, filename, dpi=600, bbox_inches='tight')
		plot_metric(d, 'f1', False)
		plot_metric(d, 'balanced_accuracy', False)
		plot_metric(d, 'f1', True)
		plot_metric(d, 'balanced_accuracy', True)