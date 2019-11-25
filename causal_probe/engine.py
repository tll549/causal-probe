import io
import numpy as np
import logging
import os
import re

from causal_probe import utils
from data import SemEval_preprocess

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from transformers import BertTokenizer, BertModel, BertForMaskedLM

PATH_TO_VEC = 'examples/glove/glove.840B.300d.txt'

SemEval_RAW_DATAPATH = 'data/causal_probing/SemEval_2010_8/raw/TRAIN_FILE.TXT'
SemEval_PROCESSED_DATAPATH = 'data/causal_probing/SemEval_2010_8/processed/SemEval_mask_processed.txt'
SemEval_LOGS_DATAPATH = 'logs/SemEval_logs'

class engine(object):
	def __init__(self, params):
		self.params = utils.dotdict(params)
		self.params.pretrained = utils.dotdict(self.params.pretrained)
		logging.info('params: ' + str(self.params))

	def eval(self):
		if self.params.reset_data:
			self.preprocess_data()
		assert os.path.exists(SemEval_PROCESSED_DATAPATH), 'should preprocess data first'
		self.load_data()
		self.prepare_data()
		self.prepare_encoder()

		if self.params.probing_task == 'simpel':
			# not done yet
			# self.encode()
			return
		elif self.params.probing_task == 'mask':
			self.predict_mask()

		self.save_pred(SemEval_LOGS_DATAPATH)

	def preprocess_data(self):
		logging.info('preprocessing data...')
		if self.params.dataset == 'semeval':
			dl = SemEval_preprocess.DataLoader()
			dl.read(SemEval_RAW_DATAPATH)
			dl.preprocess(self.params.probing_task, mask='cause')
			# dl.split(dev_prop=0.2, test_prop=0.2, seed=self.params.seed)
			dl.write(SemEval_PROCESSED_DATAPATH)

	def load_data(self):
		if self.params.dataset == 'semeval':
			fpath = 'data/causal_probing/SemEval_2010_8/processed/SemEval_mask_processed.txt'
		elif self.params.dataset == 'because':
			fpath = 'data/causal_probing/BECAUSE/processed/because_all.txt'

		self.data = {'X': [], 'y': [], 'rel': []}
		with io.open(fpath, 'r', encoding='utf-8') as f:
			for line in f:
				# line = 'te	gardens	[CLS] The winery includes [MASK]. [SEP]	Component-Whole'
				line = line.rstrip().split('\t')
				self.data['y'].append(line[1])
				self.data['X'].append(line[2])
				self.data['rel'].append(line[3])

		logging.info(f'Loaded {len(self.data["X"])}')

	def prepare_data(self):
		# print(data['X'][:3])
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
		for i in range(len(self.data['X'])):
			# print(self.data['X'][i])

			shuf = shuffle(self.data['X'][i])
			self.data['X_shuf'].append(shuf)
			# print(shuf)

			trunc = truncate(self.data['X'][i])
			self.data['X_trunc'].append(trunc)
			# print(trunc)

			shuf_trunc = truncate(shuf)
			self.data['X_shuf_trunc'].append(shuf_trunc)
			# print(shuf_trunc)
			# print()
			# break
		
		# print(self.data['X'][7555])
		# print(self.data['X_shuf'][7555])
		# print(self.data['X_trunc'][7555])
		# print(self.data['X_shuf_trunc'][7555])

	def prepare_encoder(self):
		logging.info('preparing...')
		params = self.params

		if params.pretrained.model == 'bert':
			bert_type = f'bert-{params.pretrained.model_type}-{"cased" if params.pretrained.cased else "uncased"}'
			logging.getLogger('transformers').setLevel(logging.ERROR)
			params.tokenizer = BertTokenizer.from_pretrained(bert_type)
			params.encoder = BertForMaskedLM.from_pretrained(bert_type).to(DEVICE)
			params.encoder.eval() # ??

		elif params.pretrained.model == 'glove':
			# from senteval
			samples = self.data['train']['X'] + self.data['dev']['X'] + self.data['test']['X']
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
			logging.debug(f'word2id: {params.word2id["book"]}') # 90
			logging.debug(f'word2vec: {params.word_vec["book"][:5]}') # 300

		logging.info('prepared')

	def predict_mask(self):
		logging.info('predicting...')
		k = self.params.k
		
		X_types = ['X', 'X_shuf', 'X_trunc', 'X_shuf_trunc']
		correct = {k:{X_type:[] for X_type in X_types} for k in list(set(self.data['rel']))}
		self.pred = []
		
		for i in range(len(self.data['X'])):
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
				top_inds = torch.argsort(soft_pred, descending=True)[:, :k].cpu().numpy()
				# print(top_inds.shape) # 2, 5
				# top_probs = [soft_pred[tgt_ind].item() for tgt_ind in top_inds]
				top_k_preds = [self.params.tokenizer.convert_ids_to_tokens(top_inds[to_pred, :]) \
					for to_pred in range(top_inds.shape[0])]
				# print(top_k_preds)
				# print(y)
				# print()

				num_correct = [y.split()[j] in top_k_preds[j] for j in range(len(masked_index))]
				correct[rel][X_type].append(int(all(num_correct)))

				pred += [x, top_k_preds, all(num_correct)]
			pred += [y, rel]
			print(pred)
			print()

			self.pred.append(pred)
			if self.params.trial:
				if i == 8:
					break
		self.acc = {k1:{k2:np.mean(v2) if v2 != [] else 0 for k2, v2 in v1.items()} for k1, v1 in correct.items()}
		logging.info(f'acc: {self.acc}')
		# print(self.pred[8])

	def save_pred(self, path):
		# if self.params.trial:
		# 	return

		with open(path + f'_{self.params.k}.txt', 'w+', encoding='utf-8') as f:
			for l in self.pred:
				# o = f'{l[0]}\t{str(l[1])}\t{l[2]}\t{l[3]}\t{l[4]}\n'
				o = '\t'.join([str(x) for x in l]) + '\n'
				f.write(o)
		with open(path + f'_{self.params.k}_acc.txt', 'w+', encoding='utf-8') as f:
			for k, v in self.acc.items():
				f.write(k + ' : ' + str(v) + '\n')
