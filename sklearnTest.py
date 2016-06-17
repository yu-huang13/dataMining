# -*- coding:utf-8 -*-

import os
import sys
from sklearn import feature_extraction
from sklearn.cross_validation import cross_val_score
import numpy as np

print 'test 20news---------------------'
from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
#print twenty_train
print twenty_train.target_names
print len(twenty_train.data), type(twenty_train.data), twenty_train.data[0]
print len(twenty_train.target), type(twenty_train.target), twenty_train.target
print len(twenty_train.filenames), type(twenty_train.filenames), twenty_train.filenames[0]


print 'test tfidf----------------------'
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

text = ['I love movie very much, movie!', 'I love you, yc', "I'm Huang Yu"]
count_vect = CountVectorizer(stop_words = 'english')
print '停用词：', len(count_vect.get_stop_words()), count_vect.get_stop_words()
X_train_counts = count_vect.fit_transform(text)

wordBag = X_train_counts.toarray()
word = count_vect.get_feature_names()

for i in range(len(wordBag)):
	print '---------第', i, '类文本的词频如下--------'
	for j in range(len(word)):
		print word[j], wordBag[i][j]


tf_transformer = TfidfTransformer().fit_transform(X_train_counts)
weight = tf_transformer.toarray()#将matrix 转换为ndarray

for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
	print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"
	for j in range(len(word)):
		print word[j],weight[i][j]


print 'test datasets---------------------------------'
from sklearn import datasets
digits = datasets.load_digits()
print digits.data
print digits.target, len(digits.target)
print type(digits)
print type(digits.data), len(digits.data)
print type(digits.data[0]), len(digits.data[0])
print digits.data[0]



print 'test svm---------------------------------'
from sklearn.svm import SVC
svcClf = SVC(gamma=0.001, C=100.)
svcClf.fit(digits.data[:-2], digits.target[:-2])
svmResult = svcClf.predict(digits.data[-2:])
print svmResult, type(svmResult)
print cross_val_score(svcClf, digits.data[:-2], digits.target[:-2], cv=5)


print 'test Naive Bayes------------------------'
from sklearn.naive_bayes import MultinomialNB
bayesClf = MultinomialNB()
bayesClf.fit(digits.data[:-2], digits.target[:-2])
bayesResult = bayesClf.predict(digits.data[-2:])
print bayesResult, type(bayesResult)
print cross_val_score(bayesClf, digits.data[:-2], digits.target[:-2], cv=5)


print 'test Design Tree------------------------'
from sklearn.tree import DecisionTreeClassifier
dtClf = DecisionTreeClassifier(random_state=0)
dtClf.fit(digits.data[:-2], digits.target[:-2])
dtResult = dtClf.predict(digits.data[-2:])
print dtResult
print cross_val_score(dtClf, digits.data[:-2], digits.target[:-2], cv=5)

print 'test Logistic Regression----------------'
from sklearn.linear_model import LogisticRegression
lrClf = LogisticRegression()
lrClf.fit(digits.data[:-2], digits.target[:-2])
lrResult = lrClf.predict(digits.data[-2:])
print lrResult
print cross_val_score(lrClf, digits.data[:-2], digits.target[:-2], cv=5)


print 'test Random Forest---------------------'
from sklearn.ensemble import RandomForestClassifier
rfClf = RandomForestClassifier()
rfClf.fit(digits.data[:-2], digits.target[:-2])
rfResult = rfClf.predict(digits.data[-2:])
print rfResult
print cross_val_score(rfClf, digits.data[:-2], digits.target[:-2], cv=5)

print 'test Ramdom Forest and kFolder---------'

scores = cross_val_score(RandomForestClassifier(), digits.data[:-2], digits.target[:-2], cv=5)
print scores, type(scores)

print 'test k folder--------------------------'
from sklearn.cross_validation import StratifiedKFold
X = np.array([[1, 2], [1, 4], [1, 2], [2, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(y, n_folds=2)
print len(skf), skf

for train_index, test_index in skf:
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	print 'X_train = ', X_train, ', y_train = ', y_train
	print 'X_test = ', X_test, ', y_test = ', y_test


print 'Ramdom Forest and kFolder and report-----'
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc


rfClf = RandomForestClassifier()
skf = StratifiedKFold(digits.target, n_folds = 10)
for train_index, test_index in skf:
	print '===================================='
	#print("TRAIN:", train_index, "TEST:", test_index)
	X_train, Y_train = digits.data[train_index], digits.target[train_index]
	X_test, Y_test = digits.data[test_index], digits.target[test_index]
	rfClf.fit(X_train, Y_train)
	result = rfClf.predict(X_test)
	print classification_report(Y_test, result)
	print 'accuracy = ', accuracy_score(Y_test, result)
	print 'precision_score = ', precision_score(Y_test, result)
	print 'recall_score = ', recall_score(Y_test, result)
	print 'f1_score = ', f1_score(Y_test, result)
	fpr, tpr, thresholds = roc_curve(Y_test, result, pos_label = 9)
	print 'roc_auc_score = ', auc(fpr, tpr)















