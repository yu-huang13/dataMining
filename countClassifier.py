# -*- coding:utf-8 -*-

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import os

def readFile(path, targetDict):
	print 'reading ', path
	if os.path.isfile(path):	#读取文件
		readXML(path, targetDict)
		return
	for file in os.listdir(path):	#读取文件夹
		fullPath = os.path.join(path, file)
		if os.path.isdir(fullPath):
			readFile(fullPath, targetDict)
		else:
			readXML(fullPath, targetDict)

def readXML(path, targetDict):
	if path.split('.').pop() != 'xml':
		return
	tree = ET.parse(path)
	root = tree.getroot()
	for target in getRawClassifier(root):
		if target not in targetDict:
			targetDict[target] = 1
		else:
			targetDict[target] += 1

def getRawClassifier(root):
	"""获取文本所有的分类（list of str）
	"""
	rawClassifierList = []
	for rawClassifier in root.iter('classifier'):
		if 'type' in rawClassifier.attrib and rawClassifier.attrib['type'] == 'taxonomic_classifier':
			rawClassifierList.append(rawClassifier.text)
	return rawClassifierList

path = '../nyt_corpus/data/1987'
targetDict = {}
readFile(path, targetDict)
targetList = sorted(targetDict.iteritems(), key=lambda d:d[1], reverse = True)
for target in targetList:
	print target[0], ": ", target[1]









