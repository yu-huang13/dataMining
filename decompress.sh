#!/bin/sh

#此脚本放在nyt_corpus文件夹下

cd data

for dir in `ls .`
do
	cd $dir
	for file in `ls .`
	do
		tar zxvf $file
		rm $file
	done
	cd ..
done
