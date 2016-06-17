# -*- coding:utf-8 -*-

from mylib import *
from threading import Thread
from datetime import *

print 'reading xml...'
beginTime = datetime.now()
settings = readSettings('settings.json')
stopWords = json.load(file('stopWords.json'))		#提取tf-idf矩阵较慢，是使用默认词表速度的10倍
#stopWords = 'english'
textList, targetList = readAllFile(settings)
print 'read over, using time = ', str(datetime.now() - beginTime)

print 'getting tf-idf weight matrix...'
beginTime = datetime.now()
weight = getTfIdfWeight(textList, stopWords)
print 'got tf-idf weight matrix, using time = ', str(datetime.now() - beginTime)

print 'Cross Validating...'
beginTime = datetime.now()
result = {}
threads = []
threads.append( Thread(target = runBayes, args = (result, weight, targetList, )) )
#threads.append( Thread(target = runSVM, args = (result, weight, targetList, )) )		#太慢，故删去
threads.append( Thread(target = runDecisionTree, args = (result, weight, targetList, )) )
threads.append( Thread(target = runLogisticRegression, args = (result, weight, targetList, )) )
threads.append( Thread(target = runRandomForest, args = (result, weight, targetList, )) )

for thread in threads:
	thread.start()

for thread in threads:
	Thread.join(thread)

print '--------------------------------------------------------'
print 'all result: '
for method in result.keys():
	print method + ': \n' + result[method]

print 'Cross Validate over, total using time = ', str(datetime.now() - beginTime)







