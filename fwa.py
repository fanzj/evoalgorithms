# -*- coding:utf-8 -*-
import numpy as np
import problems.evoalgorithms as op
import random
import copy
import matplotlib.pyplot as plt
import time
import math
from operator import itemgetter, attrgetter

class FireSpark:
    '''
    火星(解)
    '''
    def __init__(self,vardim,bound):
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.
    
    def generate(self):
        rnd = np.random.random(self.vardim)
        self.x = np.zeros(self.vardim)
        for i in range(0,self.vardim):
            self.x[i] = self.bound[0,i] + (self.bound[1,i] - self.bound[0,i]) * rnd[i]
    
    def calculateFitness(self):
        self.fitness = op.fitness(self.x)

    def printIndividual(self):
        print 'x: %s' % str(self.x)

class FWAAlgorithm:
    def __init__(self,sizepop,vardim,bound,maxgen,params,filename):
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.maxgen = maxgen
        self.params = params
        self.filename = filename
        self.population = [] # 种群
        self.fitness = np.zeros((self.sizepop,1))
        self.trace = np.zeros((self.maxgen,2))

        self.AMax = 40 # 爆炸幅度上限
        self.AMin = [] # 爆炸振幅下限
        self.M = 20 # 一般火星数量上限
        self.Mg = 5 # 高斯火星数量上限
        self.Ca = 0.04 # 火星数量控制参数a
        self.Cb = 0.8 # 火星数量控制参数b
        self.Pa = 0.02 # Amin的初始爆炸振幅控制参数
        self.Pb = 0.001 # Amin的最终爆炸振幅控制参数
        self.AInit = [] # Amin的初始爆炸振幅
        self.AFinal = [] # Amin的最终爆炸振幅
        self.best = None
        self.worst = None 
        self.sumDW = 0.
        self.sumDB = 0.
        self.epsilon = 0.00000000001

    def initialize(self):
        for i in range(0, self.sizepop):
            ind = FireSpark(self.vardim, self.bound)
            ind.generate()
            ind.calculateFitness()
            self.population.append(ind)
            self.fitness[i] = ind.fitness
        bestIndex = np.argmin(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        if self.Mg > self.sizepop:
            self.Mg = self.sizepop
        self.AInit = self.Pa * (self.bound[1] - self.bound[0])
        self.AFinal = self.Pb * (self.bound[1] - self.bound[0])

    def calculate(self):
        worstIndex = np.argmax(self.fitness)
        self.worst = self.population[worstIndex]
        self.sumDW = 0.
        self.sumDB = 0.
        for p in self.population:
            self.sumDW += (self.worst.fitness - p.fitness)
            self.sumDB += (p.fitness - self.best.fitness)
        self.AMin = self.AInit - (self.AInit - self.AFinal) * (math.sqrt((2 * self.maxgen - self.t) * self.t) / self.maxgen)

    def explode(self,sol):
        '''
        对解(烟花)sol进行一般爆炸
        '''
        s = (self.worst.fitness - sol.fitness + self.epsilon) / (self.sumDW + self.epsilon)
        if s < self.Ca:
            s = self.Ca
        elif s > self.Cb:
            s = self.Cb
        si = int(round(s * self.M))
        A = self.AMax * (sol.fitness - self.best.fitness + self.epsilon) / (self.sumDB + self.epsilon)
        sparks = []
        for i in range(0,si):
            sparks.append(copy.deepcopy(sol))
            bChanged = False
            for j in range(0, self.vardim):
                if random.random() < 0.5:
                    continue
                amp = self.AMin[j] if A < self.AMin[j] else A
                sparks[i].x[j] += (amp * (random.random() * 2 -1))
                if sparks[i].x[j] < self.bound[0,j] or sparks[i].x[j] > self.bound[1,j]:
                    sparks[i].x[j] = self.bound[0,j] + random.random() * (self.bound[1,j] - self.bound[0,j])
                bChanged = True
            if bChanged:
                self.evaluate(sparks[i])
        return sparks

    def explodeGauss(self,sol):
        '''
        对解(烟花)sol进行高斯爆破
        '''
        spark = copy.deepcopy(sol)
        g = random.gauss(0,1)
        bChanged = False
        for j in range(0,self.vardim):
            if random.random() < 0.5:
                continue
            spark.x[j] += (g * (self.best.x[j] - spark.x[j]))
            if spark.x[j] < self.bound[0,j] or spark.x[j] > self.bound[1,j]:
                spark.x[j] = self.bound[0,j]+ random.random() * (self.bound[1,j] - self.bound[0,j])
            bChanged = True
        if bChanged:
            self.evaluate(spark)
        return spark


    def evaluate(self,sol):
        sol.calculateFitness()
        return sol.fitness

    def saveBestMean(self):
        self.trace[self.t,0] = self.best.fitness
        for i in range(0,self.sizepop):
            self.fitness[i] = self.population[i].fitness
        self.trace[self.t,1] = np.mean(self.fitness)

    def pickBest(self,sparks):
        minIndex = 0
        minFitness = sparks[0].fitness
        for i in range(1,len(sparks)):
            if minFitness > sparks[i].fitness:
                minFitness = sparks[i].fitness
                minIndex = i
        return sparks[minIndex]

    def select(self,sparks):
        '''
        从当前烟花和火星集合中选取新的烟花种群
        '''
        pop1 = self.randomSelect(sparks,self.sizepop)
        cBest = self.pickBest(sparks)
        if cBest.fitness < self.best.fitness:
            self.best = copy.deepcopy(cBest)
        pop1[0] = copy.deepcopy(cBest)
        return pop1

    def randomSelect(self,pop,k):
        '''
        随机从种群中选择k个
        '''
        sels = []
        for i in range(0,k):
            pos = random.randint(0,len(pop)-1)
            sels.append(pop[pos])
        return sels

    def printPop(self, pop):
        for i in range(0,self.sizepop):
            pop[i].printIndividual()
   

    def printResult(self):
		'''
		plot the result of the bbo algorithm
		'''
		x = np.arange(0, self.maxgen)
		y1 = self.trace[:, 0]
		y2 = self.trace[:, 1]
		plt.plot(x,y1,'r',label='optimal value')
		plt.plot(x,y2,'g',label='average value')
		plt.xlabel("Iteration")
		plt.ylabel("function value")
		plt.title("FWA algorithm for function optimization")
		plt.legend()
		plt.show()

    
    def solve(self):
        f = open(self.filename+'.txt', 'w')
        self.t = 0
        self.initialize()
        self.saveBestMean()
        print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t,self.trace[self.t,0],self.trace[self.t,1]))
        f.write("Generation %d: optimal function value is: %f; average function value is %f\n" % (self.t,self.trace[self.t,0],self.trace[self.t,1]))
        while self.t < self.maxgen -1:
            self.t += 1
            self.calculate()
            sparks = []
            for p in self.population:
                sparks.extend(self.explode(p))
            cBest = copy.deepcopy(self.pickBest(sparks))
            if cBest.fitness < self.best.fitness:
                self.best = copy.deepcopy(cBest)
            
            sels = self.randomSelect(self.population,self.Mg)
            for i in range(0,self.Mg):
                sparks.append(self.explodeGauss(sels[i]))
            self.population = self.select(sparks)
            self.saveBestMean()

            print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t,self.trace[self.t,0],self.trace[self.t,1]))
            f.write("Generation %d: optimal function value is: %f; average function value is %f\n" % (self.t,self.trace[self.t,0],self.trace[self.t,1]))
        print("Optimal function value is: %f" % self.best.fitness)
        f.write("Optimal function value is: \n%f" % self.best.fitness)
        print("Optimal solution is:")
        print(self.best.x)
        f.write("Optimal solution is:\n")
        f.write(str(self.best.x))

        f.close


        
sizepop = 50
vardim = 15
bound = np.tile([[-3],[3]],vardim)
maxgen = 100
params = []
filename = './results/fwa_res' + time.strftime('%Y-%m-%d', time.localtime(time.time()))
fwa = FWAAlgorithm(sizepop,vardim,bound,maxgen,params,filename)
fwa.solve()
fwa.printResult()
      
