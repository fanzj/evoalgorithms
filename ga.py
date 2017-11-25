# -*- coding:utf-8 -*-
import numpy as np
import problems.evoalgorithms as op
import random
import copy
import matplotlib.pyplot as plt
import time

class GAIndividual:
	'''
	individual of genetic algorithm
	'''

	def __init__(self,vardim,bound):
		'''
		vardim: dimension of variables
		bound: boundaries of variables
		'''
		self.vardim = vardim
		self.bound = bound
		self.fitness = 0.

	def generate(self):
		'''
		generate a random chromsome for genetic algorithm
		'''
		len = self.vardim
		rnd = np.random.random(size=len)
		self.chrom = np.zeros(len)
		for i in range(0,len):
			#self.chrom[i] = random.gauss(0,1)
			self.chrom[i] = self.bound[0,i] + (self.bound[1,i] - self.bound[0,i]) * rnd[i]

	def printIndividual(self):
		'''
		print individual function
		'''
		print 'fitness: %f;' % self.fitness,'chrom:',self.chrom

	def calculateFitness(self):
		'''
		calculate the fitness of the chromsome
		'''
		self.fitness = op.fitness(self.chrom)

class GeneticAlgorithm:
	'''
	the class for genetic algorithm
	'''

	def __init__(self, sizepop, vardim, bound, maxgen, params, filename):
		'''
		sizepop: population size
		vardim: dimension of variables
		bound: boundaries of variables
		maxgen: termination condition
		params: algorithm required parameters, it is a list which is
		consisting of crossover rate, mutation rate, alpha
		params[0] is the crossover rate
        params[1] is the mutation rate
		'''
		self.sizepop = sizepop
		self.vardim = vardim
		self.bound = bound
		self.maxgen = maxgen
		self.params = params
		self.population = []
		self.fitness = np.zeros((self.sizepop,1))
		self.trace = np.zeros((self.maxgen,2))
		self.filename = filename


	def initialize(self):
		'''
		initialize the population
		'''	
		for i in range(0,self.sizepop):
			ind = GAIndividual(self.vardim, self.bound)
			ind.generate()
			self.population.append(ind)

	def evaluate(self):
		'''
		evaluation of the population fitness
		'''
		for i in range(0, self.sizepop):
			self.population[i].calculateFitness()
			self.fitness[i] = self.population[i].fitness


	def selectionOperation(self):
		'''
		selection operation for Genetic Algorithm
		roulette wheel selection
		'''
		newpop = []
		totalFitness = np.sum(self.fitness)
		accuFitness = np.zeros((self.sizepop, 1))

		accuFitness[0] = self.fitness[0] / totalFitness
		for i in range(1, self.sizepop):
			accuFitness[i] = accuFitness[i-1] + self.fitness[i] / totalFitness

		for i in range(0, self.sizepop):
			r = random.random()
			idx = 0
			for j in range(0, self.sizepop):
				if j == 0 and r < accuFitness[j]:
					idx = 0
					break
				elif r >= accuFitness[j] and r < accuFitness[j+1]:
					idx = j+1
					break
			newpop.append(self.population[idx])
		self.population = newpop

	def crossoverOperatoin(self):
		'''
		crossover operation for genetic algorithm
		'''
		newpop = []
		for i in range(0,self.sizepop,2):
			idx1 = random.randint(0, self.sizepop-1)
			idx2 = random.randint(0, self.sizepop-1)
			while idx2 == idx1:
				idx2 = random.randint(0, self.sizepop-1)
			newpop.append(copy.deepcopy(self.population[idx1]))
			newpop.append(copy.deepcopy(self.population[idx2]))
			r = random.random()
			if r < self.params[0]:
				crossPos = random.randint(0, self.vardim)
				for j in range(crossPos, self.vardim):
					newpop[i].chrom[j] = newpop[i].chrom[j] * self.params[2] + (1-self.params[2]) * newpop[i+1].chrom[j]
					newpop[i+1].chrom[j] = newpop[i+1].chrom[j] * self.params[2] + (1-self.params[2]) * newpop[i].chrom[j]
		self.population = newpop

	def mutationOperation(self):
		'''
		mutation operation for genetic algorithm
		'''
		newpop = []
		for i in range(0, self.sizepop):
			newpop.append(copy.deepcopy(self.population[i]))
			r = random.random()
			if r < self.params[1]:
				mutatePos = random.randint(0, self.vardim-1)
				theta = random.random()
				if theta > 0.5:
					newpop[i].chrom[mutatePos] = newpop[i].chrom[mutatePos] - (newpop[i].chrom[mutatePos] - self.bound[0, mutatePos]) * (1 - random.random() ** (1 - self.t / self.maxgen))
				else:
					newpop[i].chrom[mutatePos] = newpop[i].chrom[mutatePos] + (self.bound[1, mutatePos] - newpop[i].chrom[mutatePos]) * (1 - random.random() ** (1 - self.t / self.maxgen))
				#newpop[i].chrom[mutatePos] = random.uniform(bound[0][mutatePos],bound[1][mutatePos])
		self.population = newpop

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


	def printPoplulation(self):
		for i in range(self.sizepop):
			self.population[i].printIndividual()


	def solve(self):
		'''
		evolution process of genetic algorithm
		'''
		f = open(self.filename+'.txt','w')
		self.t = 0
		self.initialize()
		self.evaluate()
		best = np.min(self.fitness)
		#self.printPoplulation()
		bestIndex = np.argmin(self.fitness)
		self.best = copy.deepcopy(self.population[bestIndex])
		self.avefitness = np.mean(self.fitness)
		self.trace[self.t,0] = self.best.fitness
		self.trace[self.t,1] = self.avefitness
		print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
		f.write("Generation %d: optimal function value is: %f; average function value is %f\n" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
		while (self.t < self.maxgen - 1):
			self.t += 1
			self.selectionOperation()
			self.crossoverOperatoin()
			self.mutationOperation()
			self.evaluate()
			best = np.min(self.fitness)
			bestIndex = np.argmin(self.fitness)
			if best < self.best.fitness:
				self.best = copy.deepcopy(self.population[bestIndex])
			self.avefitness = np.mean(self.fitness)
			self.trace[self.t,0] = self.best.fitness
			self.trace[self.t,1] = self.avefitness
			print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
			f.write("Generation %d: optimal function value is: %f; average function value is %f\n" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
		print("Optimal function value is: %f" % self.trace[self.t,0])
		f.write("Optimal function value is: %f\n" % self.trace[self.t,0])
		print("Optimal solution is:")
		print(self.best.chrom)
		f.write("Optimal solution is:\n")
		f.write(str(self.best.chrom))
		f.close()

sizepop = 50
vardim = 15
bound = np.tile([[-3],[3]],vardim)
maxgen = 100
params = [0.9,0.1,0.5]
filename = './results/ga_res_' + time.strftime('%Y-%m-%d',time.localtime(time.time()))
gaAlg = GeneticAlgorithm(sizepop,vardim,bound,maxgen,params,filename)
gaAlg.solve()
gaAlg.printResult()	

