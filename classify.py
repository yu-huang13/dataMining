# -*- coding:utf-8 -*-

from mylib import *
from datetime import *

print 'reading xml...'
beginTime = datetime.now()
settings = readSettings('classify_settings.json')
#stopWords = json.load(file('longStopWords.json'))		#提取tf-idf矩阵较慢，是使用默认词表速度的10倍
stopWords = 'english'
textList, targetList = readAllFile(settings)
print 'read over, using time = ', str(datetime.now() - beginTime)

print 'Count classifer:'
for i in range(len(settings['classifier'])):
	classifier = settings['classifier'][i]
	print 'target: ', i, ', name: ', classifier, ", count: ", settings['classifierCount'][classifier]


print 'getting tf-idf weight matrix...'
beginTime = datetime.now()
weight = getTfIdfWeight(textList, stopWords)
print 'got tf-idf weight matrix, using time = ', str(datetime.now() - beginTime)


print 'Cross Validating...'
beginTime = datetime.now()
result = {}

runBayes(result, weight, targetList)
#runRandomForest(result, weight, targetList)
#runLogisticRegression(result, weight, targetList)
#runDecisionTree(result, weight, targetList)


print '--------------------------------------------------------'
print 'all result: '
for method in result.keys():
	print method + ': \n' + result[method]

print 'Cross Validate over, total using time = ', str(datetime.now() - beginTime)







