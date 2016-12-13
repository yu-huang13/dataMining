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
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import IncrementalPCA



def readSettings(filename):
	"""读取settings.json文件并进行初步处理
	Returns:
		dict: 从filename中load进来的字典
	"""
	settings = json.load(file(filename))
	settings['classifierDict'] = {}
	settings['classifierCount'] = {}
	for i in range(len(settings['classifier'])):
		settings['classifierDict'][ settings['classifier'][i] ] = i
		settings['classifierCount'][ settings['classifier'][i] ] = 0
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
		readFile(path, textList, targetList, settings['classifierDict'], settings['classifierCount'])
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
	#print '停用词表: \n', countVect.get_stop_words()		#输出停用词表
	del textList, countVect
	gc.collect()

	weight = TfidfTransformer().fit_transform(bagOfWords)
	del bagOfWords
	gc.collect()

	print 'tf-idf matrix shape = (文本数, 单词数) = ', weight.shape
	return weight


def stratifiedKFoldCrossValidation(clf, clfName, dataMatrix, targetList, nFolds = 5):
	"""k折分层交叉验证
	Args:
		clf: 分类器
		dataMatrix (scipy.sparse.csr.csr_matrix): 特征矩阵
		targetList (numpy.ndarray): 特征向量对应分类的列表
		nFolds (int): nFolds折分层交叉验证，默认为10
	Returns:
		str: 分层交叉验证的结果
	"""
	accuracy, precision, recall, f1, rocAuc = [], [], [], [], []
	skf = StratifiedKFold(targetList, n_folds = nFolds)
	drew = False
	for trainIndex, testIndex in skf:
		dataTrain, targetTrain = dataMatrix[trainIndex], targetList[trainIndex]	#训练集
		dataTest, targetTest = dataMatrix[testIndex], targetList[testIndex]		#测试集
		clf.fit(dataTrain, targetTrain)
		pred = clf.predict(dataTest)
		predProb = clf.predict_proba(dataTest)
		accuracy.append(accuracy_score(targetTest, pred))
		precision.append(precision_score(targetTest, pred))
		recall.append(recall_score(targetTest, pred))
		f1.append(f1_score(targetTest, pred))
		classiferExist = list(set(targetTest))
		rocAuc.append(calAUC(targetTest, predProb, classiferExist))
		if not drew:	#对第一折验证作ROC曲线
			drawROC('roc_' + clfName + '.jpg', targetTest, predProb, classiferExist)
			drew = True
	return 'accuracy = %f, precision = %f, recall = %f, f1 = %f, auc = %f' % (np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1), np.mean(rocAuc))	


def runCluster(clt, cltName, dataMatrix, targetList):
	"""
	Args:
		clt: 聚类器
		dataMatrix (scipy.sparse.csr.csr_matrix): 特征矩阵
		targetList (numpy.ndarray): 特征向量对应分类的列表
	Returns:
		str: 聚类的结果
	"""
	if cltName == 'Agglomerative':
		pred = clt.fit_predict(dataMatrix.toarray(), targetList)
	else:
		pred = clt.fit_predict(dataMatrix, targetList)

	"""
	print 'drawing cluster...'
	print 'changing sparse matrix to dense matrix...'
	beginTime = datetime.now()
	dataList = dataMatrix.toarray()
	print 'change over, using time =', str(datetime.now() - beginTime)

	print 'PCA feature reducing...'
	beginTime = datetime.now()
	pca = IncrementalPCA(n_components=2)
	pointList = pca.fit_transform(dataList)
	print 'reduce over, using time =', str(datetime.now() - beginTime)

	drawCluster('origin_' + cltName + '.jpg', pointList, targetList)
	drawCluster('pred_' + cltName + '.jpg', pointList, pred)"""

	adjustedMutualInformation = adjusted_mutual_info_score(targetList, pred)
	adjustedRandIndex = adjusted_rand_score(targetList, pred)
	homogeneity = homogeneity_score(targetList, pred)
	completeness = completeness_score(targetList, pred)
	vMeasure = v_measure_score(targetList, pred)
	return 'Adjusted Mutual Information = %f, Adjusted Rand Index = %f, Homogeneity = %f, Completeness = %f, V-Measure = %f' % (adjustedMutualInformation, adjustedRandIndex, homogeneity, completeness, vMeasure)


def drawCluster(filename, pointList, targetList):
	"""
	Args:
		filename (str): 保存成的文件名
		dataList (numpy.ndarray): 原tf-idf降维后的矩阵
		targetList (numpy.ndarray): 特征向量对应分类的列表
	"""
	colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#22ff77', '#ff2277', '#77ff22', '#7722ff', '#222222', '#777777']
	targetExist = set(targetList)
	for target in targetExist:
		if target >= len(colorList) - 1:
			continue
		posList = [index for index, value in enumerate(targetList) if value == target]
		X, Y = [], []
		for i in posList:
			X.append(pointList[i][0])
			Y.append(pointList[i][1])
		plt.scatter(X, Y, color = colorList[target], label = str(target))
		
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('cluster')
	plt.legend(loc='lower right')
	plt.savefig(filename)
	plt.close('all')


def calAUC(targetTest, predProb, classiferExist):
	"""遍历classiferExist作为pos_label，计算auc后汇总，返回平均值
	"""
	rocAuc = []
	for i in classiferExist:
		if i >= predProb.shape[1]:
			continue
		fpr, tpr, thresholds = roc_curve(targetTest, predProb[:, i], pos_label = i)
		rocAuc.append(auc(fpr, tpr))
	return np.mean(rocAuc)


def drawROC(filename, targetTest, predProb, classifierExist):
	"""遍历classiferExist作为pos_label，在一张图上分别绘制曲线，保存在filename中
	"""
	for i in classifierExist:
		if i >= predProb.shape[1]:
			continue
		fpr, tpr, thresholds = roc_curve(targetTest, predProb[:, i], pos_label = i)
		plt.plot(fpr, tpr, lw=1, label='classifier=%d (area=%0.2f)' % (i, auc(fpr, tpr)))
	plt.plot([0, 1], [0, 1], 'k--', color=(0.6, 0.6, 0.6))		#画对角线
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc='lower right')
	plt.savefig(filename)
	plt.close('all')


def readFile(path, dataList, targetList, classifierDict, classifierCount):
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
			readFile(fullPath, dataList, targetList, classifierDict, classifierCount)
		else:
			readXML(fullPath, dataList, targetList, classifierDict, classifierCount)


def readXML(path, dataList, targetList, classifierDict, classifierCount):
	"""读取单个xml文件
	"""
	if path.split('.').pop() != 'xml':
		return
	tree = ET.parse(path)
	root = tree.getroot()
	target, classifier = getTarget(classifierDict, getRawClassifier(root))
	if target != -1:
		targetList.append(target)
		dataList.append( handleFullText(getFullText(root)) )
		classifierCount[classifier] += 1


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
	target = -1
	belongTo = ''
	for rawClassifier in rawClassifierList:
		if not rawClassifier:
			continue
		for classifier in classifierDict.keys():
			if rawClassifier.find(classifier) != -1:
				if target != -1:	
					if target != classifierDict[classifier]:	#文本同时属于多个类，抛弃
						return -1, ''
				else:
					target = classifierDict[classifier]
					belongTo = classifier
	return target, belongTo


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


def runBayes(result, dataMatrix, targetList):
	print 'running Bayes...'
	beginTime = datetime.now()
	bayesClf = MultinomialNB(alpha = 0.01)
	result['Bayes'] = stratifiedKFoldCrossValidation(bayesClf, 'Bayes', dataMatrix, targetList)
	print '--------------------------------------------------------'
	print 'result of Bayes:\n', result['Bayes']
	print 'Bayes end, using time = ', str(datetime.now() - beginTime)


def runSVM(result, dataMatrix, targetList):
	print 'running SVM...'
	svcClf = SVC(gamma=0.001, C=100.)
	result['SVM'] = stratifiedKFoldCrossValidation(svcClf, 'SVM', dataMatrix, targetList)
	print '--------------------------------------------------------'
	print 'result of SVM:\n', result['SVM']
	print 'SVM end, using time = ', str(datetime.now() - beginTime)


def runDecisionTree(result, dataMatrix, targetList):
	print 'running Decision Tree...'
	beginTime = datetime.now()
	dtClf = DecisionTreeClassifier(random_state=0)
	result['Decision Tree'] = stratifiedKFoldCrossValidation(dtClf, 'DecisionTree', dataMatrix, targetList)
	print '--------------------------------------------------------'
	print 'result of Decision Tree:\n', result['Decision Tree']
	print 'Decision Tree end, using time = ', str(datetime.now() - beginTime)


def runLogisticRegression(result, dataMatrix, targetList):
	print 'running Logistic Regression...'
	beginTime = datetime.now()
	lrClf = LogisticRegression(C = 40.0, penalty = 'l2', solver = 'newton-cg')
	result['Logistic Regression'] = stratifiedKFoldCrossValidation(lrClf, 'LogisticRegression', dataMatrix, targetList)
	print '--------------------------------------------------------'
	print 'result of Logistic Regression:\n', result['Logistic Regression']
	print 'Logistic Regression end, using time = ', str(datetime.now() - beginTime)


def runRandomForest(result, dataMatrix, targetList):
	print 'running Random Forest...'
	beginTime = datetime.now()
	rfClf = RandomForestClassifier(n_estimators = 80, min_samples_split = 2, n_jobs = -1)
	result['Random Forest'] = stratifiedKFoldCrossValidation(rfClf, 'RandomForest', dataMatrix, targetList)
	print '--------------------------------------------------------'
	print 'result of Random Forest:\n', result['Random Forest']
	print 'Random Forest end, using time = ', str(datetime.now() - beginTime)


def runKMeans(result, dataMatrix, targetList, nClusters):
	print 'running KMeans...'
	beginTime = datetime.now()
	kmClt = KMeans(n_clusters = nClusters, max_iter = 600, tol = 1e-5, n_jobs = -1)
	result['KMeans'] = runCluster(kmClt, 'KMeans', dataMatrix, targetList)
	print '--------------------------------------------------------'
	print 'result of KMeans:\n', result['KMeans']
	print 'KMeans end, using time = ', str(datetime.now() - beginTime)


def runDBScan(result, dataMatrix, targetList, nClusters):
	"""由于稀疏矩阵的非零元过多会导致乘法溢出，故无法使用1年的数据，半年则可
	"""
	print 'running DBSCAN...'
	beginTime = datetime.now()
	dbsClt = DBSCAN(eps = 0.99, min_samples=5)

	result['DBSCAN'] = runCluster(dbsClt, 'DBSCAN', dataMatrix, targetList)
	print '--------------------------------------------------------'
	print 'result of DBSCAN:\n', result['DBSCAN']
	print 'DBSCAN end, using time = ', str(datetime.now() - beginTime)


def runAgglomerative(result, dataMatrix, targetList, nClusters):
	print 'running Agglomerative...'
	beginTime = datetime.now()
	aggClt = AgglomerativeClustering(n_clusters = nClusters)
	result['Agglomerative'] = runCluster(aggClt, 'Agglomerative', dataMatrix, targetList)
	print '--------------------------------------------------------'
	print 'result of Agglomerative:\n', result['Agglomerative']
	print 'Agglomerative end, using time = ', str(datetime.now() - beginTime)













