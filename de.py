# -*- coding:utf-8 -*-
import numpy as np
import problems.evoalgorithms as op
import random
import copy
import matplotlib.pyplot as plt
import time
import math
import tools.mathtools as tl

class DEIndividual:

    def __init__(self, vardim, bound):
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.

    def generate(self):
        rnd = np.random.random(self.vardim)
        self.x = np.zeros(self.vardim)
        for i in range(0,self.vardim):
            self.x[i] = self.bound[0,i] + (self.bound[1,i] - self.bound[0,i]) * rnd[i]

    def printIndividual(self):
        print 'fitness:',self.fitness,'x:',self.x
    
    def calculateFitness(self):
        self.fitness = op.fitness(self.x)

class DEAlgorithm:

    def __init__(self, sizepop, vardim, bound, maxgen, params, filename):
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.maxgen = maxgen
        self.params = params
        self.filename = filename
        self.population = []
        self.trace = np.zeros((self.maxgen,2))
        self.fitness = np.zeros((self.sizepop,1))
        self.best = None
        self.t = 0
        self.scalfingF = 0.5
        self.crossRate = 0.9

    def initialize(self):
        for i in range(0,self.sizepop):
            ind = DEIndividual(self.vardim, self.bound)
            ind.generate()
            ind.calculateFitness()
            self.population.append(ind)
            self.fitness[i] = ind.fitness
        bestIndex = np.argmin(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])

    def evaluate(self,ind):
        ind.calculateFitness()
        return ind.fitness

    def printPopulation(self):
        '''
        print population
        '''
        for i in range(0,self.sizepop):
            self.population[i].printIndividual()

    def saveBestMean(self):
        self.trace[self.t,0] = self.best.fitness
        for i in range(0, self.sizepop):
            self.fitness[i] = self.population[i].fitness
        self.trace[self.t,1] = np.mean(self.fitness)

    def diffEvolve(self,s,index):
        '''
        DE/rand/1
        '''
        r1,r2,r3 = tl.randomSelectThreeIndices(self.sizepop, index)
        u = copy.deepcopy(s)
        j = random.randint(0,self.vardim-1)
        for d in range(0,self.vardim):
            if random.random() < self.crossRate or j == d:
                u.x[d] = self.population[r1].x[d] + self.scalfingF * (self.population[r2].x[d] - self.population[r3].x[d])
                if u.x[d] < self.bound[0,d] or u.x[d] > self.bound[1,d]:
                    u.x[d] = self.bound[0,d] + random.random() * (self.bound[1,d] - self.bound[0,d])
        self.evaluate(u)
        return u

    def diffEvolve2(self,s,index):
        '''
        DE/rand-to-best/2
        '''
        r1,r2 = tl.randomSelectTwoIndices(self.sizepop, index)
        u = copy.deepcopy(s)
        j = random.randint(0,self.vardim-1)
        for d in range(0,self.vardim):
            if random.random() < self.crossRate or j == d:
                u.x[d] = u.x[d] + self.scalfingF * (self.best.x[d] - u.x[d]) + self.scalfingF * (self.population[r1].x[d] - self.population[r2].x[d])
                if u.x[d] < self.bound[0,d] or u.x[d] > self.bound[1,d]:
                    u.x[d] = self.bound[0,d] + random.random() * (self.bound[1,d] - self.bound[0,d])
        self.evaluate(u)
        return u




    def updateBest(self):
        bestIndex = np.argmin(self.fitness)
        if self.population[bestIndex].fitness < self.best.fitness:
            self.best = copy.deepcopy(self.population[bestIndex])

    def printResult(self):
		'''
		plot the result of the DE algorithm
		'''
		x = np.arange(0, self.maxgen)
		y1 = self.trace[:, 0]
		y2 = self.trace[:, 1]
		plt.plot(x,y1,'r',label='optimal value')
		plt.plot(x,y2,'g',label='average value')
		plt.xlabel("Iteration")
		plt.ylabel("function value")
		plt.title("DE algorithm for function optimization")
		plt.legend()
		plt.show()

    def solve(self):
        f = open(self.filename+'.txt','w')
        self.initialize()
        self.saveBestMean()
        print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
        f.write('Generation %d: optimal function value is: %f; average function value is %f\n' % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while self.t < self.maxgen - 1:
            for i in range(0,self.sizepop):
                u = self.diffEvolve2(self.population[i],i)
                if u.fitness < self.population[i].fitness:
                    self.population[i] = u
            self.t += 1
            self.saveBestMean()
            self.updateBest()
            print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t, self.trace[self.t,0], self.trace[self.t,1]))
            f.write('Generation %d: optimal function value is: %f; average function value is %f\n' % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))

        print("Optimal function value is: %f" % self.best.fitness)
        f.write("Optimal function value is: %f\n" % self.best.fitness)
        print("Optimal solution is:")
        print(self.best.x)
        f.write("Optimal solution is:\n")
        f.write(str(self.best.x))
        f.close()


sizepop = 50
vardim = 15
bound = np.tile([[-3],[3]],vardim)
maxgen = 100
params = []
filename = './results/de_res_' + time.strftime('%Y-%m-%d', time.localtime(time.time()))
de = DEAlgorithm(sizepop,vardim,bound,maxgen,params,filename)
de.solve()
de.printResult()