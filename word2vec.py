"""
create the word2vec model
resouce come from http://www.sogou.com/labs/

author: Simon
date: 31/8/2016
"""
import jieba
import gensim
import os


def get_stop_words():

	# load stop words from file

	stop_word_set = set()

	with open('stop_words') as f:
		for word in f.readlines():
			stop_word_set.add(word.strip())

	return stop_word_set


def sen2words(sentence,stop_word_set):

	# split words from sentence with jieba
	# split the stop words

	words_list = []

	seg_words = jieba.lcut(sentence,cut_all=False)

	for word in seg_words:
		if word not in stop_word_set:
			words_list.append(word)

	return words_list

class MySentences(object):

	# iter load the data

	def __init__(self,filename,stop_word_set):
	 	self.filename= filename
	 	self.stop_word_set = stop_word_set

	# def __iter__(self):

	# 	files =[f for f in os.listdir(self.files_dir) if not f.endswith('_sub')]

	# 	for file in files:
	# 		with open(self.files_dir+'/'+file,'r',encoding='utf-8',errors='ignore') as f:
	# 			for line in f.readlines():
	# 				yield sen2words(line,self.stop_word_set)

	def __iter__(self):

		with open(self.filename,'r',encoding='utf-8',errors='ignore') as f:
			for line in f.readlines():
				yield sen2words(line, self.stop_word_set)

def train_save(filename,modelname):
	stop_word_set = get_stop_words()
	sentences = MySentences(filename,stop_word_set)

	num_features = 20
	min_word_count = 20
	#num_workers = 48
	#context = 20
	#spoch = 20
	#sample = 1e-5
	model = gensim.models.Word2Vec(
		sentences,
		size = num_features,
		min_count = min_word_count,
		#workers = num_workers,
		#sample = sample,
		#window = context,
		#iter = spoch,
		)

	model.save(modelname)

	return model



