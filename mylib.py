# -*- coding:utf-8 -*-

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import os
import sys
import gc
import json
import re
import numpy as np
from datetime import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def readSettings(filename):
	"""读取settings.json文件并进行初步处理
	Returns:
		dict: 从filename中load进来的字典
	"""
	settings = json.load(file(filename))
	settings['classifierDict'] = {}
	for i in range(len(settings['classifier'])):
		settings['classifierDict'][ settings['classifier'][i] ] = i
	return settings


def readAllFile(settings):
	"""读取所有指定目录下的所有xml文件
	Args:
		settings (dict): readSettings()函数中load进来的字典
	Returns:
		numpy.ndarray, numpy.ndarray: 文本列表，分类列表
	"""
	textList = []
	targetList = []
	pathList = settings['path']
	for path in pathList:
		readFile(path, textList, targetList, settings['classifierDict'])
	return np.array(textList), np.array(targetList)


def getTfIdfWeight(textList, stopWords = 'english'):
	"""获取tf-idf权重矩阵
	Args:
		textList (list of str): 所有输入的文本，textList[i]表示第i个文本的内容
		stopWords (str or list): 停用词列表
    Returns:
    	scipy.sparse.csr.csr_matrix: weight[i][j]表示单词j在文本i中的tf-idf权重，行为文本，列为单词
	"""
	countVect = CountVectorizer(stop_words = stopWords)		
	bagOfWords = countVect.fit_transform(textList)
	del textList, countVect
	gc.collect()

	weight = TfidfTransformer().fit_transform(bagOfWords)
	del bagOfWords
	gc.collect()

	print 'tf-idf matrix shape = (文本数, 单词数) = ', weight.shape
	return weight


def stratifiedKFoldCrossValidation(clf, dataList, targetList, nFolds = 5):
	"""k折分层交叉验证
	Args:
		clf: 分类器
		dataList (scipy.sparse.csr.csr_matrix): 特征矩阵
		targetList (numpy.ndarray): 特征向量对应分类的列表
		nFolds (int): nFolds折分层交叉验证，默认为10
	Returns:
		str: 分层交叉验证的结果
	"""
	accuracy, precision, recall, f1 = [], [], [], []
	skf = StratifiedKFold(targetList, n_folds = nFolds)
	for trainIndex, testIndex in skf:
		dataTrain, targetTrain = dataList[trainIndex], targetList[trainIndex]	#训练集
		dataTest, targetTest = dataList[testIndex], targetList[testIndex]		#测试集
		clf.fit(dataTrain, targetTrain)
		pred = clf.predict(dataTest)
		accuracy.append(accuracy_score(targetTest, pred))
		precision.append(precision_score(targetTest, pred))
		recall.append(recall_score(targetTest, pred))
		f1.append(f1_score(targetTest, pred))
	assert len(accuracy) == nFolds
	return 'accuracy = %f, precision = %f, recall = %f, f1 = %f' % (np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1))	



def readFile(path, dataList, targetList, classifierDict):
	"""递归读取path下的所有xml文件
	Args:
		path (str): 文件或文件夹路径
		dataList (list of str): 文本列表，传进引用，用于接收读取的文本
		targetList (list of int): 分类列表，传进引用，用于接收读取的文本分类
		classifierDict (dict): classifierDict[str]代表str类对应的int值
	"""
	print 'reading ', path
	if os.path.isfile(path):	#读取文件
		readXML(path, dataList, targetList, classifierDict)
		return
	for file in os.listdir(path):	#读取文件夹
		fullPath = os.path.join(path, file)
		if os.path.isdir(fullPath):
			readFile(fullPath, dataList, targetList, classifierDict)
		else:
			readXML(fullPath, dataList, targetList, classifierDict)


def readXML(path, dataList, targetList, classifierDict):
	"""读取单个xml文件
	"""
	if path.split('.').pop() != 'xml':
		return
	tree = ET.parse(path)
	root = tree.getroot()
	target = getTarget(classifierDict, getRawClassifier(root))
	if target != -1:
		targetList.append(target)
		dataList.append( handleFullText(getFullText(root)) )


def getRawClassifier(root):
	"""获取文本所有的分类（list of str）
	"""
	rawClassifierList = []
	for rawClassifier in root.iter('classifier'):
		if 'type' in rawClassifier.attrib and rawClassifier.attrib['type'] == 'taxonomic_classifier':
			rawClassifierList.append(rawClassifier.text)
	return rawClassifierList


def getTarget(classifierDict, rawClassifierList):
	"""获取文本分类（int）
	"""
	for rawClassifier in rawClassifierList:
		if (not rawClassifier) or rawClassifier == '':
			return -1
		for classifier in classifierDict.keys():
			if rawClassifier.find(classifier) != -1:
				return classifierDict[classifier]
	return -1


def getFullText(root):
	"""获得文本
	"""
	fullText = ''
	for block in root.iter('block'):
		if 'class' in block.attrib and block.attrib['class'] == 'full_text':
			for p in block:
				fullText += ' ' + p.text
	return fullText


def handleFullText(fullText):
	"""处理获取的文本
	"""
	regex = re.compile('[^a-z0-9]')		#将字母数字外的字符转换为空格
	fullText = regex.sub(' ', fullText)
	regex = re.compile('[0-9]+')		#将数字替换为#
	fullText = regex.sub('#', fullText)
	return fullText


def runBayes(result, dataList, targetList):
	print 'running Bayes...'
	beginTime = datetime.now()
	bayesClf = MultinomialNB()
	result['Bayes'] = stratifiedKFoldCrossValidation(bayesClf, dataList, targetList)
	print '--------------------------------------------------------'
	print 'result of Bayes:\n', result['Bayes']
	print 'Bayes end, using time = ', str(datetime.now() - beginTime)


def runSVM(result, dataList, targetList):
	print 'running SVM...'
	svcClf = SVC(gamma=0.001, C=100.)
	result['SVM'] = stratifiedKFoldCrossValidation(svcClf, dataList, targetList)
	print '--------------------------------------------------------'
	print 'result of SVM:\n', result['SVM']
	print 'SVM end, using time = ', str(datetime.now() - beginTime)


def runDecisionTree(result, dataList, targetList):
	print 'running Decision Tree...'
	beginTime = datetime.now()
	dtClf = DecisionTreeClassifier(random_state=0)
	result['Decision Tree'] = stratifiedKFoldCrossValidation(dtClf, dataList, targetList)
	print '--------------------------------------------------------'
	print 'result of Decision Tree:\n', result['Decision Tree']
	print 'Decision Tree end, using time = ', str(datetime.now() - beginTime)


def runLogisticRegression(result, dataList, targetList):
	print 'running Logistic Regression...'
	beginTime = datetime.now()
	lrClf = LogisticRegression()
	result['Logistic Regression'] = stratifiedKFoldCrossValidation(lrClf, dataList, targetList)
	print '--------------------------------------------------------'
	print 'result of Logistic Regression:\n', result['Logistic Regression']
	print 'Logistic Regression end, using time = ', str(datetime.now() - beginTime)


def runRandomForest(result, dataList, targetList):
	print 'running Random Forest...'
	beginTime = datetime.now()
	rfClf = RandomForestClassifier(n_jobs = 8)
	result['Random Forest'] = stratifiedKFoldCrossValidation(rfClf, dataList, targetList)
	print '--------------------------------------------------------'
	print 'result of Random Forest:\n', result['Random Forest']
	print 'Random Forest end, using time = ', str(datetime.now() - beginTime)





