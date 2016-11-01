# —*- coding:utf-8 -*-

import gensim
import numpy
import jieba
import os
import sklearn
import numpy

from sklearn.linear_model.logistic import LogisticRegression 

def load():
	t_string1 = '标准间太差 房间还不如3星的 而且设施非常陈旧.建议酒店把老的标准间从新改善.'
	t_string2 = '标间不好，房间不如三星的，设施老化，希望重新装修。'

	t_string3 = '商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!'
	t_string4 = '房间很宽敞，经济，便宜，性价比高挺好'

	t_string5 = '地点很方便，房间很舒服，服务也很好，就是价格不便宜啊！'
	t_string6 = '地点便利，房间舒服，服务到位，价格贵'

	t_string7 = '情节太过简单。简介内容中说“作者从少年人的角度洞悉人性的温情，通过男女主人公的所见所感，细腻道出所有人对亲情、友情和爱情幸福的向往。巧妙、惊人的情节交错，让这部小说别具一格，值得瞩目”，看完之后觉得完全不是那么回事。另外，这篇小说是绝对不能媲美《追风筝的人》，无论是内容的深刻还是情节的丰富，根本不在一个数量级上。'
	t_string8 = '简直无语了，看了前几节，越看越觉得我是不是下载了错了书，这故事情节跟这么高大上的书名对不上啊，还专门上百度查了一下原书的情节，我没看错，简直就是个坑，那么虚有其表的宣传语不知道是谁写的，还跟《追风筝的人》等相提并论，真是敢说！再也不相信所谓的畅销书了。那么高的评价是怎么来的，愿书不要再这么商业化了。'

	t_string9 = '生活中不正常的事物与艺术的关系是正常的，而且是生活中与艺术保持正常关系的唯一事物。'
	t_string10 = '我喜欢人甚于原则，此外我还喜欢没原则的人甚于世界上的一切。'

def count_distance_single(st1,st2):

	singlemodel=gensim.models.Word2Vec.load('news_model_single')

	la,lb = [],[]

	for s in st1:
		if s in singlemodel:
			la.append(singlemodel[s])

	for s in st2:
		if s in singlemodel:
			lb.append(singlemodel[s])

	cos = cos_distance(list(count_vec(la)),list(count_vec(lb)))

	return cos

def count_vec_single(seg):

	ll = []
	singlemodel=gensim.models.Word2Vec.load('news_model_single')

	for s in seg:
		if s in singlemodel:
			ll.append(singlemodel[s])

	vec = count_vec(ll)

	return vec


def count_distance_word(st1,st2):

	wordmodel = gensim.models.Word2Vec.load('news_model_words')
	la,lb = [],[]

	_la = jieba.lcut(st1,cut_all=False)
	_lb = jieba.lcut(st2,cut_all=False)

	for w in _la:
		if w in wordmodel:
			la.append(wordmodel[w])
	for w in _lb:
		if w in wordmodel:
			lb.append(wordmodel[w])

	cos = cos_distance(list(count_vec(la)),list(count_vec(lb)))

	return cos

def count_vec_word(seg):
	ll = []
	wordmodel = gensim.models.Word2Vec.load('news_model_words')

	_ll = jieba.lcut(seg,cut_all=False)

	for w in _ll:
		if w in wordmodel:
			ll.append(wordmodel[w])

	vec = count_vec(ll)

	return vec

def cos_distance(list_a,list_b):
	
	# count cos_distance
	b1,b2,b3,b = 0,0,0,0
	if len(list_a) == len(list_b):
		for i in range(len(list_a)):
			b1 += float(list_a[i])**2 
			b2 += float(list_b[i])**2
			b3 += float(list_a[i])*float(list_b[i])
			b = (b1**(0.5))*(b2**(0.5))
		#print(list_a)
	else:
		print(" what's wrong with you? ")
	return b3/b


def count_vec(array_list):

	num = numpy.array([0.0]*20)
	count = 0

	for ar in array_list:
		num += ar
		count += 1
	
	return num/count

def load_train_set(dirname1,dirname2):

	pl = [i for i in os.listdir(dirname1) if not i.startswith('.')]
	nl = [i for i in os.listdir(dirname2) if not i.startswith('.')]

	vec_single_list_p,vec_word_list_p,vec_single_list_n,vec_word_list_n = [],[],[],[]

	for seg in pl:
		with open(dirname1+'/'+seg,encoding='gbk') as f:
			try:
				line = f.readline()
				vec_single_list_p.append(count_vec_single(line))
				vec_word_list_p.append(count_vec_word(line))
			except:
				print(seg+" can't decode")

	for eg in nl:
		with open(dirname2+'/'+eg,encoding='gbk') as f:
			try:
				line = f.readline()
				vec_single_list_n.append(count_vec_single(line))
				vec_word_list_n.append(count_vec_word(line))
			except:
				print(eg+" can't decode")

	length1,length2 = len(vec_single_list_p),len(vec_word_list_n)
	label = [1]*length1+[-1]*length2


	return numpy.array(vec_single_list_p+vec_single_list_n),numpy.array(vec_word_list_p+vec_word_list_n),numpy.array(label)

def train(dirname1,dirname2,dirname3,dirname4):

	s_list_train,w_list_train,label_train = load_train_set(dirname1, dirname2)
	s_list_test,w_list_test,label_test = load_train_set(dirname3, dirname4)

	classifer_single = LogisticRegression()
	classifer_word = LogisticRegression()

	classifer_single.fit(s_list_train,label_train)
	classifer_word.fit(w_list_train,label_train)

	predictions_s = classifer_single.predict(s_list_test)
	predictions_w = classifer_word.predict(w_list_test)


	length = len(predictions_s)
	count1,count2 = 0,0

	for i in range(length):
		if predictions_s[i] == label_test[i]:
			count1 += 1
		if predictions_w[i] == label_test[i]:
			count2 += 1

	print("the accuracy of classifier for single: " + str(count1/length)+"\nthe accuracy of classifier for word: "+str(count2/length))

	return predictions_s,predictions_w,label_train,label_test
if __name__ == '__main__':

	load()
	count_distance.count_distance(t_string1,t_string8)







