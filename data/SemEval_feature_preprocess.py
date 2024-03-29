import re
import logging
import random
import numpy as np
import pandas as pd

from causal_probe import utils

from zipfile import ZipFile
import codecs
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import bigrams, trigrams

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

	def preprocess(self, trial=False, swap_cause_effect=False):
		all_relations_dict = {k:re.sub(r'\(e\d,e\d\)', '', k) for k in list(set(self.y))}

		X, y = [], []
		sub, obj = [], []
		for i in range(len(self.X)):
			self.rel.append(all_relations_dict[self.y[i]])

			e1_pattern, e2_pattern = r'<e1>(.*)</e1>', r'<e2>(.*)</e2>'
			e1 = re.findall(e1_pattern, self.X[i])[0]
			e2 = re.findall(e2_pattern, self.X[i])[0]
			sub_temp = e1 if '(e1,e2)' in self.y[i] else e2
			obj_temp = e2 if '(e1,e2)' in self.y[i] else e1
			sub.append(sub_temp)
			obj.append(obj_temp)

			if swap_cause_effect:
				x = re.sub(e1_pattern, obj_temp if '(e1,e2)' in self.y[i] else sub_temp, self.X[i])
				x = re.sub(e2_pattern, sub_temp if '(e1,e2)' in self.y[i] else obj_temp, x)[1:-1]
			else:
				x = re.sub(r'</?e\d>', '', self.X[i])[1:-1]

			x = re.sub(r"(.) 's ", r"\1's ", x) # one's becomes one 's in original dataset
			x = re.sub(r" '(.*?)'", r' "\1"', x) # originally semeval use 'sth' instead of "sth", chage it back but don't touch
			
			X.append(x)

			if trial:
				if i >= 9:
					break

		self.X = X
		self.sub, self.obj = sub, obj

		logging.info(f'processed len X: {len(self.sub)}, causal: {sum([rel == "Cause-Effect" for rel in self.rel])}')

		self.output = pd.DataFrame({'X': self.X, 
			'causal': [l == 'Cause-Effect' for l in self.rel]})

	def calc_prob(self):
		def tokenize(X):
			'''split and lower'''
			handled_punct = [re.sub(r'([.,!?;])', r' \1', x) for x in X]
			return [[x2.lower() for x2 in x.split()] for x in handled_punct]
		tok = tokenize(self.X)
		self.num_sent = len(self.X)

		logging.info(f'calculating features...')
		self.output = pd.DataFrame(columns=[
			'X', 'relation', 'cause', 'effect', 
			'c_count', 'e_count', 'c_e_count', 'e_no_c_count', 'causal_dependency',
			'P(E|C)', 'P(E)', 'probabilistic_causality', 'probabilistic_causality_diff',
			'delta_P', 'P(E|no C)', 'q', 'p', 'causal_power'])
		for i in range(len(self.X)):
			c, e = self.sub[i],  self.obj[i]

			# causal dependency
			c_count = sum([c in sent for sent in self.X])
			e_count = sum([e in sent for sent in self.X])
			c_e_count = sum([c in sent and e in sent for sent in self.X])
			e_no_c_count = e_count - c_e_count # sum([e in sent and c not in sent for sent in self.X])
			causal_dependency = c_count == c_e_count and e_no_c_count == 0 # P(E|C) = 1 and P(E|not C) = 0

			# probabilistic causality
			P_of_E_given_C = c_e_count / c_count
			P_of_E = e_count / self.num_sent
			probabilistic_causality = P_of_E_given_C >= P_of_E
			probabilistic_causality_diff = P_of_E_given_C - P_of_E
			# assert probabilistic_causality_diff > 0, f'{P_of_E_given_C}, {P_of_E}, {c_e_count}, {c_count}, {e_count}'

			# delta P
			P_E_given_no_C = e_no_c_count / (self.num_sent - c_count)
			delta_P = P_of_E_given_C - P_E_given_no_C

			# causal power
			q = delta_P / (1 - P_E_given_no_C)
			if P_E_given_no_C == 0:
				p = 0
			else:
				p = -delta_P / P_E_given_no_C
			causal_power = q - p

			self.output.loc[i, :] = [self.X[i], self.rel[i], c, e, 
				c_count, e_count, c_e_count, e_no_c_count, causal_dependency,
				P_of_E_given_C, P_of_E, probabilistic_causality, probabilistic_causality_diff,
				delta_P, P_E_given_no_C, q, p, causal_power]
		logging.info(f'features calculated for {self.output.shape[0]} sentences')

	def calc_prob_oanc(self, oanc_datapath, use_semeval_first=True, trial=False):
		logging.getLogger('numexpr').setLevel(logging.ERROR)

		if use_semeval_first: # can avoid c_count = 0 in oanc
			self.calc_prob()
		else:
			self.output = pd.DataFrame({'X': self.X, 'relation': self.rel, 'cause': self.sub, 'effect': self.obj})
			for c in ['c_count', 'e_count', 'c_e_count', 'e_no_c_count']:
				self.output[c] = 0
			self.num_sent = 0

		logging.info(f'calculating features using OANC...')
		fdist_uni, fdist_bi, fdist_tri = FreqDist(), FreqDist(), FreqDist()
		with ZipFile(oanc_datapath) as zf:
			txt_written_files = [fn for fn in zf.namelist() if '.txt' in fn and 'written' in fn]
			logging.info(f'there are {len(txt_written_files)} txt written files')
			pb = utils.ProgressBar(len(txt_written_files))
			for f_i, f_path in enumerate(txt_written_files):
				pb.now(f_i)
				all_lines = ''
				with zf.open(f_path) as f:
					for line in codecs.iterdecode(f, 'utf8'):
						all_lines += line
				all_lines = re.sub(r'\s\s+', ' ', all_lines) # remove repeated spaces ' +'
				
				# sep and process by line
				sentences = nltk.sent_tokenize(all_lines)
				for sent in sentences:
					tok = [w.lower() for w in word_tokenize(sent)]
					self.num_sent += 1
					c_in = self.output.cause.str.lower().isin(tok)
					e_in = self.output.effect.str.lower().isin(tok)
					self.output.c_count += c_in
					self.output.e_count += e_in
					self.output.c_e_count += c_in & e_in
					self.output.e_no_c_count += ~c_in & e_in

				# unigram, bigram, and trigram for calc avg freq
				for word in tok:
					fdist_uni[word] += 1
				for bi in bigrams(tok):
					fdist_bi[bi] += 1
				for tri in trigrams(tok):
					fdist_tri[bi] += 1

				if trial:
					if f_i > 10:
						break
		logging.info(f'iterated through OANC, {self.num_sent} sentences')

		# causal dependency
		self.output['causal_dependency'] = (self.output.c_count == self.output.c_e_count) & (self.output.e_no_c_count == 0)
		# probabilistic causality
		self.output['P(E|C)'] = self.output.c_e_count / self.output.c_count
		self.output['P(E)'] = self.output.e_count / self.num_sent
		self.output['probabilistic_causality'] = self.output['P(E|C)'] >= self.output['P(E)']
		self.output['probabilistic_causality_diff'] = self.output['P(E|C)'] - self.output['P(E)']
		# delta P
		self.output['P(E|no C)'] = self.output.e_no_c_count / (self.num_sent - self.output.c_count)
		self.output['delta_P'] = self.output['P(E|C)'] - self.output['P(E|no C)']

		# causal power
		self.output['q'] = self.output.delta_P / (1 - self.output['P(E|no C)'])
		self.output['p'] = (-self.output.delta_P / self.output['P(E|no C)'].replace({0 : np.nan})).fillna(0) # handle divide by 0
		self.output['causal_power'] = self.output.q - self.output.p

		# PMI
		# D11-1027, Do et al., 2017
		self.N = fdist_uni.N()
		self.output['PMI'] = np.log(self.output.c_e_count.astype(int) * self.N / (self.output.c_count.astype(int) * self.output.e_count.astype(int))) # don't know why dtype is obj

		# PPMI, CPMI, NPMI, NNEGPMI
		# Salle & Villavicencio, 2019
		self.output.loc[self.output.PMI >= 0, 'PPMI'] = self.output.loc[self.output.PMI >= 0, 'PMI']
		self.output.loc[self.output.PMI < 0, 'PPMI'] = 0

		self.output.loc[self.output.PMI >= -2, 'CPMI_-2'] = self.output.loc[self.output.PMI >= -2, 'PMI']
		self.output.loc[self.output.PMI < -2, 'CPMI_-2'] = -2

		self.output['NPMI'] = self.output['PMI'] / -np.log(self.output.c_e_count.astype(int) / self.N)

		self.output.loc[self.output['PMI'] >= 0, 'NNEGPMI'] = self.output.loc[self.output['PMI'] >= 0, 'PMI']
		self.output.loc[self.output['PMI'] < 0, 'NNEGPMI'] = self.output.loc[self.output['PMI'] < 0, 'NPMI']

		# causal strength
		# Luo et al., 2016
		alpha = 0.66
		self.output['P(C|E)'] = self.output.c_e_count / self.output.e_count
		self.output['causal_stength_nec'] = (self.output['P(C|E)'] / self.N) / (self.output.c_count / self.N) ** alpha
		self.output['causal_stength_suf'] = (self.output['P(E|C)'] / self.N) / (self.output.e_count / self.N) ** alpha
		lambda_cs_list = [0.5, 0.7, 0.9, 1.0]
		for lambda_cs in lambda_cs_list:
			self.output[f'causal_stength_{lambda_cs}'] = self.output.causal_stength_nec ** lambda_cs * self.output.causal_stength_suf ** (1 - lambda_cs)

		# avg frequency, overall frequency
		def calc_avg_freq(s, fdist):
			return np.mean([fdist[w] for w in s]) / fdist.N()
		def calc_ovr_freq(s, fdist):
			return np.sum(np.log([fdist[w] / fdist.N() for w in s]))
		X_unigram = self.output.X.apply(lambda x: [w.lower() for w in word_tokenize(x)])
		X_bigram = X_unigram.apply(lambda x: list(bigrams(x)))
		X_trigram = X_unigram.apply(lambda x: list(trigrams(x)))
		self.output['avg_freq_uni'] = X_unigram.apply(calc_avg_freq, args=(fdist_uni, ))
		self.output['avg_freq_bi'] = X_bigram.apply(calc_avg_freq, args=(fdist_bi, ))
		self.output['avg_freq_tri'] = X_trigram.apply(calc_avg_freq, args=(fdist_tri, ))
		self.output['ovr_freq_uni'] = X_unigram.apply(calc_ovr_freq, args=(fdist_uni, ))
		self.output['ovr_freq_bi'] = X_bigram.apply(calc_ovr_freq, args=(fdist_bi, ))
		self.output['ovr_freq_tri'] = X_trigram.apply(calc_ovr_freq, args=(fdist_tri, ))

		if trial:
			pd.set_option('display.max_columns', 1000)
			print(self.output.head())

	def make_categorical(self, num_classes, num_classes_by):
		'''make each numerical variables in each relation categorical'''
		def float_categorize(s, num_classes, by):
			'''s should be numerical pd.series'''
			if by == 'linear':  
				s = ((s - s.min()) / ((s.max() - s.min()) / num_classes)).astype(int)
				s[s == num_classes] = num_classes - 1
				return s
			elif by == 'quantile':
				return pd.qcut(s, num_classes, labels=False)
		numerical_columns = [c for c in self.output.columns[8:] if self.output[c].nunique() > num_classes]
		for c in numerical_columns:
			for rel in self.output.relation.unique():
				try:
					self.output.loc[self.output.relation==rel, c + '_cat'] = \
						float_categorize(pd.to_numeric(self.output.loc[self.output.relation==rel, c]), 
							num_classes, num_classes_by)
				except ValueError:
					print(f'{c}, {rel}, this combination will be dropped later, ValueError: Bin edges must be unique')
				except KeyError:
					print(f'{c}, {rel} KeyError')
				# assert self.output[c].nunique() <= num_classes, f'more than {num_classes} classes'

	def save_output(self, data_path):
		utils.save_dt(self.output, data_path, index=False)
