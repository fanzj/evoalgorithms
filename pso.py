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
		self.pbest = None # history best

	def generate(self):
		'''
		generate a random particle for particle swarm optimization algorithm
		'''
		rnd = np.random.random(self.vardim)
		self.x = np.zeros(self.vardim)
		self.v = np.zeros(self.vardim)
		for i in range(0, self.vardim):
			self.x[i] = self.bound[0,i] + (self.bound[1,i] - self.bound[0,i]) * rnd[i]
			#self.v[i] = self.x[i]
			self.v[i] = 0.5 * (random.random() * (self.bound[1,i] - self.bound[0,i]) - self.x[i])

	def printIndividual(self):
		'''
		print individual function
		'''
		print 'fitness:',self.fitness,'x:',self.x

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
		self.trace = np.zeros((self.maxgen,2))
		self.pbest = []

	def initialize(self):
		'''
		initialize the population
		'''
		fit = []
		for i in range(0, self.sizepop):
			ind = PSOIndividual(self.vardim, self.bound)
			ind.generate()
			ind.calculateFitness()
			ind.pbest = copy.deepcopy(ind) # history best
			self.population.append(ind)
			fit.append(ind.fitness)
		gbestIndex = np.argmin(fit)
		self.gbest = copy.deepcopy(self.population[gbestIndex])


	def evaluate(self,sol):
		'''
		evaluation of single solution fitness
		'''
		sol.calculateFitness()
		return sol.fitness
			
		# self.fitness[i] = self.population[i].fitness

	def updateW(self):
		'''
		update inertia cofficient
		'''
		self.w = self.params[0] -  (self.params[0] - self.params[1]) * self.t * 1.0 / self.maxgen

	def move(self,sol):
		vj = []
		xj = []
		for j in range(0,self.vardim):
			r1 = random.random()
			r2 = random.random()
			new_vj = self.w * sol.v[j] + self.params[2] * r1 * (sol.pbest.x[j] - sol.x[j]) + self.params[3] * r2 * (self.gbest.x[j] - sol.x[j])
			vj.append(new_vj)

			new_xj = sol.x[j] + new_vj
			if new_xj > self.bound[1,j]:
				# new_xj = self.bound[1,j]
				new_xj = self.bound[0,j] + random.random() * (self.bound[1,j] - self.bound[0,j])
				vj[j] = 0
			elif new_xj < self.bound[0,j]:
				#new_xj = self.bound[0,j]
				new_xj = self.bound[0,j] + random.random() * (self.bound[1,j] - self.bound[0,j])
				vj[j] = 0
			xj.append(new_xj)
		sol.v = vj
		sol.x = xj

		self.evaluate(sol)
		if sol.fitness < sol.pbest.fitness:
			sol.pbest = copy.deepcopy(sol)
			if sol.fitness < self.gbest.fitness:
				self.gbest = copy.deepcopy(sol)

		return sol

	
	def printResult(self):
		'''
		plot the result of the particle swarm optimization algorithm
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

	def saveBestMean(self):
		self.trace[self.t,0] = self.gbest.fitness
		fit = []
		for i in range(0, self.sizepop):
			fit.append(self.population[i].fitness)
		self.trace[self.t,1] = np.mean(fit)


	def solve(self):
		'''
		evolution process of pso
		'''
		f = open(self.filename+'.txt','w')
		self.t = 0
		self.w = params[0]
		self.initialize()

		self.saveBestMean()
		print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
		f.write('Generation %d: optimal function value is: %f; average function value is %f\n' % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
		while self.t < self.maxgen-1:
			self.updateW()
			for i in range(0,self.sizepop):
				self.move(self.population[i])
			self.t += 1
			self.saveBestMean()
			print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
			f.write('Generation %d: optimal function value is: %f; average function value is %f\n' % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))

		print("Optimal function value is: %f" % self.gbest.fitness)
		f.write("Optimal function value is: %f\n" % self.gbest.fitness)
		print("Optimal solution is:")
		print(self.gbest.x)
		f.write("Optimal solution is:\n")
		f.write(str(self.gbest.x))
		f.close()

		
		# 记录pso当前时间每次运行的最优适应度
		f2 = open(op.getOptimalPath('pso')+'.txt','a+')
		# 写入配置信息
		f2.write('sizepop: %d; dimension: %d; maxgen: %d\n' % (self.sizepop, self.vardim, self.maxgen))
		f2.write('wmax: %f; wmin: %f; c1: %f; c2: %f\n' % (self.params[0], self.params[1], self.params[2], self.params[3]))
		f2.write("Optimal function value is: %f\n\n" % self.gbest.fitness)
		f2.close()


sizepop = 50
vardim = 15
bound = np.tile([[-3],[3]],vardim)
maxgen = 100
params = [0.9,0.4,1.49618,1.49618]
# params = [0.9, 0.4, 2.0, 2.0]
filename = './results/pso_res_' + time.strftime('%Y-%m-%d', time.localtime(time.time()))
pso = PSOAlgorithm(sizepop,vardim,bound,maxgen,params,filename)
pso.solve()
pso.printResult()




