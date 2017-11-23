# -*- coding:utf-8 -*-
import numpy as np
import problems.evoalgorithms as op
import random
import copy
import matplotlib.pyplot as plt
import time

class PSOIndividual:
	'''
	individual of particle swarm optimization algorithm
	'''
	def __init__(self, vardim, bound):
		'''
		vardim: dimension of variables
		bound: boundaries of variables
		'''
		self.vardim = vardim
		self.bound = bound
		self.fitness = 0.

	def generate(self):
		'''
		generate a random particle for particle swarm optimization algorithm
		'''
		rnd = np.random.random(self.vardim)
		self.x = np.zeros(self.vardim)
		self.v = np.zeros(self.vardim)
		for i in range(0, self.vardim):
			self.x[i] = self.bound[0,i] + (self.bound[1,i] - self.bound[0,i]) * rnd[i]
			self.v[i] = self.x[i]

	def printIndividual(self):
		'''
		print individual function
		'''
		print 'fitness:',self.fitness,'x:',x

	def calculateFitness(self):
		'''
		calculate the fitness of particle
		'''
		self.fitness = op.fitness(self.x)

class PSOAlgorithm:
	'''
	the class for pso
	'''
	def __init__(self, sizepop, vardim, bound, maxgen, params, filename):
		'''
		sizepop: population size
		vardim: dimension of variables
		bound: boundaries of variables x, v ([xmin,xmax,vmin,vmax])
		maxgen: termination condition
		params: algorithm required parameters, it is a list which is consisting of inertia coefficient,
		learning factors (wmax = params[0], wmin = params[1], c1 = params[2], c2 = params[3])
		filename: reuslt saving
		'''
		self.sizepop = sizepop
		self.vardim = vardim
		self.bound = bound
		self.maxgen = maxgen
		self.params = params
		self.filename = filename
		self.population = []
		self.fitness = np.zeros((self.sizepop,1))
		self.trace = np.zeros((self.maxgen,2))
		self.pbest = []

	def initialize(self):
		'''
		initialize the population
		'''
		for i in range(0, self.sizepop):
			ind = PSOIndividual(self.vardim, self.bound)
			ind.generate()
			self.population.append(ind)

	def evaluate(self):
		'''
		evaluation of population fitness
		'''
		for i in range(0, self.sizepop):
			self.population[i].calculateFitness()
			self.fitness[i] = self.population[i].fitness

	def printResult(self):
		'''
		plot the result of the genetic algorithm
		'''
		x = np.arange(0, self.maxgen)
		y1 = self.trace[:, 0]
		y2 = self.trace[:, 1]
		plt.plot(x,y1,'r',label='optimal value')
		plt.plot(x,y2,'g',label='average value')
		plt.xlabel("Iteration")
		plt.ylabel("function value")
		plt.title("Particle swarm optimization algorithm for function optimization")
		plt.legend()
		plt.show()


	def printPoplulation(self):
		for i in range(self.sizepop):
			self.population[i].printIndividual()

	def updateW(self):
		'''
		update inertia cofficient
		'''
		self.w = self.params[0] -  (self.params[0] - self.params[1]) * self.t / self.maxgen

	def updateXV(self):
		for i in range(0,self.sizepop):
			vj = []
			xj = []
			for j in range(0,self.vardim):
				r1 = random.random()
				r2 = random.random()
				new_vj = self.w * self.population[i].v[j] + self.params[2] * r1 * (self.pbest[i].x[j] - self.population[i].x[j]) + self.params[3] * r2 * (self.gbest.x[j] - self.population[i].x[j])
				if new_vj > self.bound[3,j]:
					new_vj = self.bound[3,j]
				elif new_vj < self.bound[2,j]:
					new_vj = self.bound[2,j]

				new_xj = self.population[i].x[j] + new_vj
				if new_xj > self.bound[1,j]:
					new_xj = self.bound[1,j]
				elif new_xj < self.bound[0,j]:
					new_xj = self.bound[0,j]

				vj.append(new_vj)
				xj.append(new_xj)
			self.population[i].v = vj
			self.population[i].x = xj
		

	def initBest(self):
		gbestIndex = np.argmin(self.fitness)
		self.gbest = copy.deepcopy(self.population[gbestIndex]) # 全局最优粒子
		for i in range(0, sizepop):
			self.pbest.append(copy.deepcopy(self.population[i])) # 历史最优粒子

	def updateBest(self):
		for k in range(0,self.sizepop):
			if self.population[k].fitness < self.pbest[k].fitness:
				self.pbest[k] = copy.deepcopy(self.population[k])
			if self.pbest[k].fitness < self.gbest.fitness:
				self.gbest = copy.deepcopy(self.pbest[k])

	def printResult(self):
		'''
		plot the result of the genetic algorithm
		'''
		x = np.arange(0, self.maxgen)
		y1 = self.trace[:, 0]
		y2 = self.trace[:, 1]
		plt.plot(x,y1,'r',label='optimal value')
		plt.plot(x,y2,'g',label='average value')
		plt.xlabel("Iteration")
		plt.ylabel("function value")
		plt.title("Genetic algorithm for function optimization")
		plt.legend()
		plt.show()



	def solve(self):
		'''
		evolution process of pso
		'''
		f = open(self.filename+'.txt','w')
		self.t = 0
		self.w = params[0]
		self.initialize()
		self.evaluate()
		self.initBest()
		self.trace[self.t,0] = self.gbest.fitness
		self.trace[self.t,1] = np.mean(self.fitness)
		print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
		f.write('Generation %d: optimal function value is: %f; average function value is %f\n' % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
		while self.t < self.maxgen-1:
			self.t += 1
			self.updateXV()
			self.evaluate()
			self.updateBest()
			self.updateW()

			self.trace[self.t,0] = self.gbest.fitness
			#self.trace[self.t,1] = np.mean(self.fitness)
			fit = []
			for i in range(0, self.sizepop):
				fit.append(self.pbest[i].fitness)
			self.trace[self.t,1] = np.mean(fit)
			print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
			f.write('Generation %d: optimal function value is: %f; average function value is %f\n' % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
		print("Optimal function value is: %f" % self.gbest.fitness)
		f.write("Optimal function value is: %f\n" % self.gbest.fitness)
		print("Optimal solution is:")
		print(self.gbest.x)
		f.write("Optimal solution is:\n")
		f.write(str(self.gbest.x))
		f.close()




		


sizepop = 50
vardim = 15
bound = np.tile([[-3],[3],[-1],[1]],vardim)
maxgen = 100
params = [0.9,0.4,1.49618,1.49618]
filename = './results/pso_res_' + time.strftime('%Y-%m-%d', time.localtime(time.time()))
pso = PSOAlgorithm(sizepop,vardim,bound,maxgen,params,filename)
pso.solve()
pso.printResult()





