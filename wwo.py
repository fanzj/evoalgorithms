# -*- coding:utf-8 -*-
import numpy as np
import problems.evoalgorithms as op
import random
import copy
import matplotlib.pyplot as plt
import time
import math
import sys

class WaveIndividual:
	def __init__(self,vardim,bound,waveheight):
		self.vardim = vardim
		self.bound = bound
		self.waveheight = waveheight
		self.wavelen = 0.5
		self.fitness = 0.


	def generate(self):
		rnd = np.random.random(self.vardim)
		self.wave = np.zeros(self.vardim)
		for i in range(0,self.vardim):
			self.wave[i] = self.bound[0,i] + (self.bound[1,i] - self.bound[0,i]) * rnd[i]

	def calculateFitness(self):
		self.fitness = op.fitness(self.wave)

	def printIndividual(self):
		print 'fitness:',self.fitness,'wave:',self.wave

class WWOAlgorithm:
	def __init__(self,vardim,bound,maxgen,params,filename):
		'''
		sizepop: population size (sizepop[0] = min, sizepop[1] = max)
		vardim: dimension of variables
		bound: boundaries of variables
		maxgen: termination condition
		params: algorithm required parameters, it is a list 
		(params[0] = hmax, params[1] = alpha, params[2] = beta min, params[3] = beta max, params[4] = min population size,
		params[5] = max population size) 
		'''
		self.vardim = vardim
		self.bound = bound
		self.maxgen = maxgen
		self.params = params
		self.filename = filename
		self.population = []
		self.sizepop = params[5]
		self.fitness = np.zeros((self.sizepop,1))
		self.trace = np.zeros((self.maxgen,2))
		self.beta = params[3]
		self.epsilon = 0.0001
		self.kmax = min(12, self.vardim / 2)
		self.dimrange = []
		for i in range(0,self.vardim):
			self.dimrange.append(self.bound[1,i] - self.bound[0,i])

	def initialize(self):
		for i in range(0,self.sizepop):
			ind = WaveIndividual(self.vardim,self.bound,self.params[0])
			ind.generate()
			self.population.append(ind)

	def evaluateAll(self):
		for i in range(0, self.sizepop):
			self.evaluate(i)

	def evaluate(self, waveIndex):
		self.population[waveIndex].calculateFitness()
		self.fitness[waveIndex] = self.population[waveIndex].fitness

	def initBestWorst(self):
		bestIndex = np.argmin(self.fitness)
		self.best = copy.deepcopy(self.population[bestIndex])
		worstIndex = np.argmax(self.fitness)
		self.worst = copy.deepcopy(self.population[worstIndex])

	def updateBestWorst(self):
		for i in range(0, self.sizepop):
			if self.population[i].fitness < self.best.fitness:
				self.best = copy.deepcopy(self.population[i])
			if self.population[i].fitness > self.worst.fitness:
				self.worst = copy.deepcopy(self.population[i])

	def updateSizepop(self):
		'''
		population size gradually decrease
		'''
		self.sizepop = self.params[4] - (self.params[5] - self.params[4]) * self.t / self.maxgen

	def updateBeta(self):
		'''
		update the cofficient of beta
		'''
		self.beta = self.params[3] - (self.params[3] - self.params[2]) * self.t / self.maxgen

	def updateWavelen(self):
		'''
		update the wave length
		'''
		for i in range(0, self.sizepop):
			new_wavelen = self.population[i].wavelen
			new_wavelen *= math.pow(self.params[1], -(self.population[i].fitness - self.worst.fitness + self.epsilon) / (self.best.fitness - self.worst.fitness + self.epsilon))
			self.population[i].wavelen = new_wavelen

	def propagate(self, waveIndex):
		'''
		传播算子
		'''
		new_wavelen = self.population[waveIndex].wavelen
		new_wave = self.population[waveIndex].wave
		for j in range(0,self.vardim):
			r = random.uniform(-1,1)
			temp = new_wave[j] + r * new_wavelen * self.dimrange[j]
			if temp > self.bound[1, j]:
				temp = self.bound[1,j]
			if temp < self.bound[0,j]:
				temp = self.bound[0,j]

		self.population[waveIndex].wave = new_wave
		self.evaluate(waveIndex)

	def refract(self, waveIndex):
		'''
		折射算子
		'''
		originWave = copy.deepcopy(self.population[waveIndex])
		new_wave = self.population[waveIndex].wave
		new_bestwave = self.best.wave
		for j in range(0,self.vardim):
			'''
			高斯分布 N(u,g)
			'''
			u = (new_wave[j] + new_bestwave[j]) / 2.0
			g = abs(new_bestwave[j] - new_wave[j]) / 2.0
			r = random.gauss(u, math.sqrt(g))
			if r > self.bound[1,j]:
				r = self.bound[1,j]
			if r < self.bound[0,j]:
				r = self.bound[0,j]

			new_wave[j] = r

		self.population[waveIndex].wave = new_wave
		self.evaluate(waveIndex)
		self.population[waveIndex].waveheight = self.params[0]

		r = originWave.fitness / self.population[waveIndex].fitness
		if r < 1: 
			'''
			折射后，波长减小，说明解更优，更新
			'''
			w = originWave.wavelen * r
			self.population[waveIndex].wavelen = w
		else:
			'''
			保留原有解，不做操作
			'''
			self.population[waveIndex] = originWave
		return self.population[waveIndex]


	def breaking(self, waveIndex):
		'''
		碎浪算子
		'''
		kbest = WaveIndividual(self.vardim,self.bound,self.params[0])
		kbest.generate()
		kbest.fitness = sys.maxint * 1.0

		k = random.randint(1,self.kmax)
		for i in range(0,k+1):
			new_wave = copy.deepcopy(self.population[i])
			dimpos = random.randint(self.vardim - 1)
			wave_x = new_wave.wave
			temp = wave_x[dimpos] + random.gauss(0,1) * self.beta * self.dimrange[dimpos]

			if temp > self.bound[1, dimpos]:
				temp = self.bound[1, dimpos]
			if temp < self.bound[0, dimpos]:
				temp = self.bound[0, dimpos]

			wave_x[dimpos] = temp
			new_wave.wave = wave_x
			new_wave.calculateFitness()

			if new_wave.fitness < kbest.fitness:
				kbest = new_wave

		if kbest.fitness < self.best.fitness:
			'''
			最优的独立波比最优的波好
			'''
			self.best = kbest


	def evoluation(self):
		'''
		进化 整合传播、折射、碎浪
		'''
		for i in range(0,self.sizepop):
			tempWave = copy.deepcopy(self.population[i]) # 未作修改
			self.propagate(i)
			if self.population[i].fitness < tempWave.fitness:
				self.population[i].waveheight = self.params[0]
				if self.population[i].fitness < self.best.fitness:
					self.best = copy.deepcopy(self.population[i])
					self.breaking(self.population[i])
				tempWave = self.population[i]
			else:
				h = tempWave.waveheight
				h -= 1
				tempWave.waveheight = h
				if h == 0:
					tempWave = self.refract(i)
			self.population[i] = copy.deepcopy(tempWave)

	def solve(self):
		f = open(self.filename+'.txt','w')
		self.t = 0
		self.initialize()
		self.evaluateAll()
		self.initBestWorst()
		self.trace[self.t,0] = self.best.fitness
		self.trace[self.t,1] = np.mean(self.fitness)
		print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
		f.write('Generation %d: optimal function value is: %f; average function value is %f\n' % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
		while self.t < self.maxgen -1:
			self.t += 1
			self.evoluation()
			self.updateWavelen()
			self.updateBestWorst()
			self.updateBeta()
			# self.updateSizepop()

			self.trace[self.t,0] = self.best.fitness
			self.trace[self.t,1] = np.mean(self.fitness)
			print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
			f.write('Generation %d: optimal function value is: %f; average function value is %f\n' % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
		print("Optimal function value is: %f" % self.best.fitness)
		f.write("Optimal function value is: %f\n" % self.best.fitness)
		print("Optimal solution is:")
		print(self.best.wave)
		f.write("Optimal solution is:\n")
		f.write(str(self.best.wave))
		f.close()


			

vardim = 15
bound = np.tile([[-3],[3]],vardim)
maxgen = 100
params = [12,1.0026,0.001,0.25,3,50]
filename = './results/wwo_res_' + time.strftime('%Y-%m-%d',time.localtime(time.time()))
wwo = WWOAlgorithm(vardim,bound,maxgen,params,filename)
wwo.solve()

