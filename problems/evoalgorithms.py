
import math
import random

def fitness(chrom):
	chrom_len = len(chrom)
	sum = 0.
	for i in range(0,chrom_len):
		for k in range(0,21):
			sum += (0.5 ** k * math.cos(3 ** k * math.pi * chrom[i] + 0.5) + math.sin(5 ** k * math.pi * chrom[i]))
	return sum

'''
chrom = []
for i in range(0,10):
	chrom.append(random.gauss(0,1))

print fitness(chrom)
'''