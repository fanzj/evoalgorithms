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
	def __init__(self,vardim,bound,amplitude):
		self.vardim = vardim
		self.bound = bound
		self.amplitude = amplitude # 振幅 （波长）
		self.wavelen = 0.5 # 波长
		self.fitness = 0.


	def generate(self):
		rnd = np.random.random(self.vardim)
		self.x = np.zeros(self.vardim)
		for i in range(0,self.vardim):
			self.x[i] = self.bound[0,i] + (self.bound[1,i] - self.bound[0,i]) * rnd[i]

	def calculateFitness(self):
		self.fitness = op.fitness(self.x)

	def printIndividual(self):
		print 'fitness:',self.fitness,'x:',self.x

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
        self.trace = np.zeros((self.maxgen,2))
        self.fitness = np.zeros((self.sizepop,1))
        self.beta = params[3] # 初始碎浪范围系数
        self.btimes = params[6] # 碎浪个数
        self.hmax = params[0] # 初始波高（振幅）
        self.alpha = params[1] # 波长递减基数
        self.epsilon = 0.0001
		#self.kmax = min(12, self.vardim / 2)
        self.dimrange = [] # 搜索范围
        for i in range(0,self.vardim):
			self.dimrange.append(self.bound[1,i] - self.bound[0,i])
        self.dimsrange = [self.beta * x for x in self.dimrange] # 碎浪范围
	
        self.best = None # 当前种群的最优解
        self.worst = None # 当前种群的最差解

    def initialize(self):
		for i in range(0,self.sizepop):
			ind = WaveIndividual(self.vardim,self.bound,self.hmax)
			ind.generate()
			ind.calculateFitness()
			self.population.append(ind)
			self.fitness[i] = ind.fitness
		self.initBestWorst()

    def initBestWorst(self):
		bestIndex = np.argmin(self.fitness)
		self.best = copy.deepcopy(self.population[bestIndex])
		worstIndex = np.argmax(self.fitness)
		self.worst = copy.deepcopy(self.population[worstIndex])

    def evaluate(self, wave):
		wave.calculateFitness()
		return wave.fitness

    def updateBestWorst(self):
		for i in range(0, self.sizepop):
			if self.population[i].fitness < self.best.fitness:
				self.best = copy.deepcopy(self.population[i])
			if self.population[i].fitness > self.worst.fitness:
				self.worst = copy.deepcopy(self.population[i])
	
    def calculate(self):
		for i in range(0, self.sizepop):
			new_wavelen = self.population[i].wavelen
			r = math.pow(self.alpha, -(self.population[i].fitness - self.worst.fitness + self.epsilon) / (self.best.fitness - self.worst.fitness + self.epsilon))
			self.population[i].wavelen *= r
		self.beta = self.params[3] - (self.params[3] - self.params[2]) * self.t / self.maxgen
		self.dimsrange = [self.beta * x for x in self.dimrange]

    def updateSizepop(self):
		'''
		population size gradually decrease
		'''
		self.sizepop = self.params[5] - (self.params[5] - self.params[4]) * self.t / self.maxgen

    def propagate(self, w,index):
		'''
		传播
		'''
		w1 = copy.deepcopy(w)
		for d in range(0,self.vardim):
			w1.x[d] += (random.random() * 2 - 1) * w1.wavelen * self.dimrange[d]
			if w1.x[d] < self.bound[0,d] or w1.x[d] > self.bound[1,d]:
				w1.x[d] = self.bound[0,d] + random.random() * (self.bound[1,d] - self.bound[0,d])
		self.evaluate(w1)
		w1.amplitude = self.hmax
		return w1

    def refract(self, w, index):
		'''
		折射
		'''
		w1 = copy.deepcopy(w)
		for d in range(0,self.vardim):
			u = (w.x[d] + self.best.x[d]) / 2.0
			g = abs(self.best.x[d] - w.x[d]) / 2.0
			w1.x[d] = random.gauss(u, math.sqrt(g))
			if w1.x[d] < self.bound[0,d] or w1.x[d] > self.bound[1,d]:
				w1.x[d] = self.bound[0,d] + random.random() * (self.bound[1,d] - self.bound[0,d])
		self.evaluate(w1)
		w1.amplitude = self.hmax
		w1.wavelen = w.wavelen * w1.fitness / w.fitness
		return w1


    def breaking(self, w):
		'''
		碎浪
		'''
		w1 = copy.deepcopy(w)
		d = random.randint(0,self.vardim-1)
		w1.x[d] += random.gauss(0,1) * self.dimsrange[d]
		if w1.x[d] < self.bound[0,d] or w1.x[d] > self.bound[1,d]:
			w1.x[d] = self.bound[0,d] + random.random() * (self.bound[1,d] - self.bound[0,d])
		self.evaluate(w1)
		if w1.fitness < w.fitness:
			w1.amplitude = self.hmax
			w1.wavelen  = w.wavelen * w1.fitness / w.fitness
			if self.beta > 0.003:
				self.beta *= 0.9999
				for i in range(0,self.vardim):
					self.dimsrange[i] = self.beta * self.dimrange[i]
			return w1
		return w

    def multibreak(self,w,times):
		'''
		多次碎浪
		'''
		for i in range(0,times):
			w1 = self.breaking(w)
			if w1.fitness < w.fitness:
				w = w1
		return w

    def continuebreak(self,w):
		w1 = self.breaking(w)
		while w1.fitness < w.fitness:
			w = w1
			w1 = self.breaking(w1)
		return w

    def surge(self,w,index):
		w1 = self.propagate(w,index)
		# print 'w1.fitness: %f; w.fitness: %f' % (w1.fitness, w.fitness)
		if w1.fitness < w.fitness:
			if w1.fitness < self.best.fitness:
				times = min(round(self.btimes * self.best.fitness) / w1.fitness, self.vardim/2)
				w1 = self.multibreak(w1,times)
				self.best = copy.deepcopy(w1)
			return w1
		else:
			w.amplitude -= 1
			if w.amplitude == 0 and w.fitness > self.best.fitness:
				w = self.refract(w,index)
				if w.fitness < self.best.fitness:
					self.best = copy.deepcopy(w)
			return w

    def saveBestMean(self):
		self.trace[self.t,0] = self.best.fitness
		fit = []
		for i in range(0, self.sizepop):
			fit.append(self.population[i].fitness)
		self.trace[self.t,1] = np.mean(fit)

    def printResult(self):
		'''
		plot the result of the water wave optimization algorithm
		'''
		x = np.arange(0, self.maxgen)
		y1 = self.trace[:, 0]
		y2 = self.trace[:, 1]
		plt.plot(x,y1,'r',label='optimal value')
		plt.plot(x,y2,'g',label='average value')
		plt.xlabel("Iteration")
		plt.ylabel("function value")
		plt.title("Water wave optimization algorithm for function optimization")
		plt.legend()
		plt.show()

    def solve(self):
		f = open(self.filename+'.txt','w')
		self.t = 0
		self.initialize()
		self.saveBestMean()
		print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
		f.write('Generation %d: optimal function value is: %f; average function value is %f\n' % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
		while self.t < self.maxgen -1:
			self.t += 1

			for i in range(0,self.sizepop):
				self.population[i] = self.surge(self.population[i],i)
			self.updateBestWorst()
			self.calculate()
			#self.updateSizepop()
			self.saveBestMean()
			#print self.t,":",self.sizepop
			print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
			f.write('Generation %d: optimal function value is: %f; average function value is %f\n' % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
		print("Optimal function value is: %f" % self.best.fitness)
		f.write("Optimal function value is: %f\n" % self.best.fitness)
		print("Optimal solution is:")
		print(self.best.x)
		f.write("Optimal solution is:\n")
		f.write(str(self.best.x))
		f.close()


			

vardim = 15
bound = np.tile([[-3],[3]],vardim)
maxgen = 100
params = [12,1.0026,0.001,0.25,3,50,12]
filename = './results/wwo_res_' + time.strftime('%Y-%m-%d',time.localtime(time.time()))
wwo = WWOAlgorithm(vardim,bound,maxgen,params,filename)
wwo.solve()
wwo.printResult()


