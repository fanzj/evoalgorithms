# -*- coding:utf-8 -*-
import math
import random
import time

def fitness(chrom):
	chrom_len = len(chrom)
	sum = 0.
	for i in range(0,chrom_len):
		for k in range(0,21):
			sum += (0.5 ** k * math.cos(3 ** k * math.pi * chrom[i] + 0.5) + math.sin(5 ** k * math.pi * chrom[i]))
	return sum

def getOptimalPath(algType):
	return './results/optimal/' + algType + '_' + time.strftime('%Y-%m-%d',time.localtime(time.time()))
