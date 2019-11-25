import io
import numpy as np
import logging
import os

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
		self.prepare()

		if self.params.probing_task == 'simpel':
			# not done yet
			self.encode()

		elif self.params.probing_task == 'mask':
			self.predict_mask()

		self.save_pred(SemEval_LOGS_DATAPATH)
		return

	def preprocess_data(self):
		if self.params.dataset == 'semeval':
			dl = SemEval_preprocess.DataLoader()
			dl.read(SemEval_RAW_DATAPATH)
			dl.preprocess(self.params.probing_task, mask='cause')
			dl.split(dev_prop=0.2, test_prop=0.2, seed=self.params.seed)
			dl.write(SemEval_PROCESSED_DATAPATH)

	def load_data(self):
		if self.params.dataset == 'semeval':
			fpath = 'data/causal_probing/SemEval_2010_8/processed/SemEval_mask_processed.txt'
		elif self.params.dataset == 'because':
			fpath = 'data/causal_probing/BECAUSE/processed/because_all.txt'

		# from senteval
		self.tok2split = {'tr': 'train', 'va': 'dev', 'te': 'test'}
		self.data = {'train': {'X': [], 'y': []},
						  'dev': {'X': [], 'y': []},
						  'test': {'X': [], 'y': []}}
		with io.open(fpath, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.rstrip().split('\t')
				self.data[self.tok2split[line[0]]]['X'].append(line[-1])
				self.data[self.tok2split[line[0]]]['y'].append(line[1])

		# labels = sorted(np.unique(self.data['train']['y']))
		# self.tok2label = dict(zip(labels, range(len(labels))))
		# self.nclasses = len(self.tok2label)

		# for split in self.data:
		# 	for i, y in enumerate(self.data[split]['y']):
		# 		self.data[split]['y'][i] = self.tok2label[y]

		logging.info('Loaded %s train - %s dev - %s test' %
					 (len(self.data['train']['y']), len(self.data['dev']['y']),
					  len(self.data['test']['y'])))

	def prepare(self):
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
		# tokenize
		k = self.params.k
		X = self.data['train']['X'] + self.data['train']['X'] + self.data['train']['X']
		y = self.data['train']['y'] + self.data['train']['y'] + self.data['train']['y']
		correct = 0
		self.pred = []
		for i in range(len(X)):
			# print(X[i])
			tokenized_text = self.params.tokenizer.tokenize(X[i])
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
			top_tok_preds = [self.params.tokenizer.convert_ids_to_tokens(top_inds[to_pred, :]) \
				for to_pred in range(top_inds.shape[0])]
			# print(top_tok_preds)
			# print(y[i])
			# print()

			num_correct = [y[i].split()[j] in top_tok_preds[j] for j in range(len(masked_index))]
			if all(num_correct):
				correct += 1

			self.pred.append([X[i], top_tok_preds, y[i], num_correct])

			# if i == 8:
			# 	break
		acc = correct / len(y)
		logging.info(f'acc: {acc}')

	def save_pred(self, path):
		with open(path + f'_{self.params.k}.txt', 'w+', encoding='utf-8') as f:
			for l in self.pred:
				o = f'{l[0]}\t{str(l[1])}\t{l[2]}\t{l[3]}\n'
				f.write(o)
