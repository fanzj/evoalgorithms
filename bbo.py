# -*- coding:utf-8 -*-
import numpy as np
import problems.evoalgorithms as op
import random
import copy
import matplotlib.pyplot as plt
import time
from operator import itemgetter, attrgetter

class Habitat:
    '''
    栖息地(解)
    '''
    def __init__(self,vardim,bound):
        '''
        lam = params[0]: 迁入率
        mu = params[1]: 迁出率
        prob = params[2]: 变异率
        默认值均为0.
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.
        self.lam = 0.
        self.mu = 0.
        self.prob = 0.

    def generate(self):
        rnd = np.random.random(self.vardim)
        self.x = np.zeros(self.vardim)
        for i in range(0,self.vardim):
            self.x[i] = self.bound[0,i] + (self.bound[1,i] - self.bound[0,i]) * rnd[i]
    
    def calculateFitness(self):
        self.fitness = op.fitness(self.x)

    def printIndividual(self):
        print 'fitness: %f; lam: %f; mu: %f; prob: %f' % (self.fitness, self.lam, self.mu, self.prob)
        print 'x: %s' % str(self.x)

class BBOAlgorithm:
    def __init__(self,sizepop,vardim,bound,maxgen,params,filename):
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.maxgen = maxgen
        self.params = params
        self.filename = filename
        self.population = [] # 种群
        self.fitness = np.zeros((self.sizepop,1))
        #self.fitness = np.zeros(self.sizepop)
        self.trace = np.zeros((self.maxgen,2))

        self.maxImig = 1. # 最大迁入率
        self.maxEmig = 1. # 最大迁出率
        self.rmut = 0.01 # 最大变异率
        self.probs = np.zeros(self.sizepop)
        self.probMax = 0. # 最大变异率

        self.muSum = 0. # 种群迁出率之和
        self.cBest = None # 当前种群的最优解
        self.cWorst = None # 当前种群的最差解
        self.best = None # 全局最优解
        self.epsilon = 0.00000000001
        self.perform = np.zeros(3) 

    def initialize(self):
        for i in range(0,self.sizepop):
            ind = Habitat(self.vardim,self.bound)
            ind.generate()
            ind.calculateFitness()
            self.population.append(ind)
        self.population = sorted(self.population, key=attrgetter('fitness'))
        self.cBest = self.population[0]
        self.cWorst = self.population[self.sizepop-1]
        self.best = copy.deepcopy(self.population[0])
        self.getProbs()
        self.probMax = max(self.probs)
        #print self.probMax


    def getProbs(self):
        self.probs[0] = 1.0
        upper = self.sizepop
        below = 1.0
        for i in range(1,self.sizepop / 2 + 1):
            upper /= 4.0
            self.probs[i] = upper / below
            for j in range(0,i):
                self.probs[j] /= 4.0
            upper *= (self.sizepop - i)
            below *= (i + 1)
        for i in range(self.sizepop / 2 + 1,self.sizepop):
            self.probs[i] = self.probs[self.sizepop - i]
        # print self.probs

    def evaluate(self,sol):
        sol.calculateFitness()
        return sol.fitness

    def calculate(self):
        '''
        计算种群中每个个体的迁入、迁出、变异率
        '''
        self.muSum = 0.
        den = self.cWorst.fitness - self.cBest.fitness + self.epsilon
        for i in range(0,self.sizepop):
            self.population[i].lam = self.maxImig * (self.population[i].fitness - self.cBest.fitness + self.epsilon) / den
            self.population[i].mu = self.maxEmig * (self.cWorst.fitness - self.population[i].fitness + self.epsilon) / den
            self.muSum += self.population[i].mu
            self.population[i].prob = self.rmut * (1.0 - self.probs[i] / self.probMax)

    def migrate(self,h,index):
        '''
        迁移
        '''
        bChanged = False
        u = copy.deepcopy(h)
        for d in range(0,self.vardim):
            if random.random() < h.lam:
                selectSol = self.rouletteSelect(h) # 轮盘赌选择
                u.x[d] = selectSol.x[d]
                #print 'self.t:%d; index:%d; d:%d; muSum:%f' % (self.t,index,d,self.muSum)
                if bChanged == False:
                    bChanged = True
        if bChanged:
            self.evaluate(u)
        return u

    def rouletteSelect(self, h):
        muList = np.zeros((self.sizepop,1)) # 种群中各个个体的迁出率
        for i in range(0,self.sizepop):
            muList[i] = self.population[i].mu
        totalMu = np.sum(muList)
        accuMu = np.zeros((self.sizepop,1))

        accuMu[0] = muList[0] / totalMu
        for i in range(1,self.sizepop):
            accuMu[i] = accuMu[i-1] + muList[i]/totalMu

        r = random.random()
        selected = 0
        for i in range(0,self.sizepop):
            if i == 0 and r < accuMu[i]:
                selected = 0
                break
            elif r < accuMu[i]:
                selected = i
                break
        return self.population[selected]

    def mutate(self, h, index):
        '''
        变异
        '''
        bChanged = False
        u = copy.deepcopy(h)
        for d in range(0, self.vardim):
            if random.random() < h.prob:
                u.x[d] = self.bound[0,d] + random.random() * (self.bound[1,d] - self.bound[0,d])
                if bChanged == False:
                    bChanged = True
        if bChanged:
            self.evaluate(u)
        return u

    def updateBest(self):
        self.cBest = self.population[0]
        self.cWorst = self.population[self.sizepop - 1]
        #print 't: %d; cBest: %f; cWorst: %f; best: %f' % (self.t, self.cBest.fitness, self.cWorst.fitness, self.best.fitness)
        if self.cBest.fitness < self.best.fitness:
            self.best = copy.deepcopy(self.cBest)
            self.perform[0] += 1
        elif self.cBest.fitness > self.best.fitness:
            '''
            elite
            '''
            for i in range(self.sizepop-1,0,-1):
                self.population[i] = self.population[i-1]
            self.cBest = copy.deepcopy(self.best)
            self.population[0] = self.cBest
            self.cWorst = copy.deepcopy(self.population[self.sizepop-1])
            self.perform[1] += 1
        else:
            self.perform[2] += 1

    def saveBestMean(self):
        self.trace[self.t,0] = self.best.fitness
        for i in range(0,self.sizepop):
            self.fitness[i] = self.population[i].fitness
        self.trace[self.t,1] = np.mean(self.fitness)


    def printPopulation(self):
        for i in range(self.sizepop):
            self.population[i].printIndividual()

    def printBest(self):
        print 'best: %f; cBest: %f; cWorst: %f' % (self.best.fitness, self.cBest.fitness, self.cWorst.fitness)

    def printPop(self, pop):
        for i in range(0,self.sizepop):
            pop[i].printIndividual()
    
    def testSortPopulationByFitness(self):
        '''
        对种群按适应度值进行排序的测试函数
        '''
        print '种群排序前:'
        self.printPopulation()
        self.population = sorted(self.population, key=attrgetter('fitness'))
        print '种群排序后:'
        self.printPopulation()

    def isSameObject(self):
        '''
        判断对象是否引用相等
        '''
        bound = np.tile([[-3],[3]],vardim)
        a = Habitat(3, bound)
        b = Habitat(3, bound)
        c = b
        print id(a)
        print id(b)
        print id(c) == id(a)

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
		plt.title("BBO algorithm for function optimization")
		plt.legend()
		plt.show()

    
    def solve(self):
        f = open(self.filename+'.txt', 'w')
        self.t = 0
        self.initialize()
        self.saveBestMean()
        self.calculate()
        print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t,self.trace[self.t,0],self.trace[self.t,1]))
        f.write("Generation %d: optimal function value is: %f; average function value is %f\n" % (self.t,self.trace[self.t,0],self.trace[self.t,1]))
        while self.t < self.maxgen - 1:
            self.t += 1
            pop1 = []
            for i in range(0,self.sizepop):
                pop1.append(self.migrate(self.population[i],i))
            pop1 = sorted(pop1,key=attrgetter('fitness'))
            for i in range(self.sizepop / 2, self.sizepop):
                pop1[i] = self.mutate(self.population[i],i)
            pop1 = sorted(pop1, key=attrgetter('fitness'))
            self.population = pop1
            self.updateBest()
            self.saveBestMean()
            self.calculate()
            print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t,self.trace[self.t,0],self.trace[self.t,1]))
            f.write("Generation %d: optimal function value is: %f; average function value is %f\n" % (self.t,self.trace[self.t,0],self.trace[self.t,1]))
        print("Optimal function value is: %f" % self.best.fitness)
        f.write("Optimal function value is: \n%f" % self.best.fitness)
        print("Optimal solution is:")
        print(self.best.x)
        f.write("Optimal solution is:\n")
        f.write(str(self.best.x))

        print self.perform
        f.close


        
sizepop = 50
vardim = 15
bound = np.tile([[-3],[3]],vardim)
maxgen = 100
params = []
filename = './results/bbo_res' + time.strftime('%Y-%m-%d', time.localtime(time.time()))
bbo = BBOAlgorithm(sizepop,vardim,bound,maxgen,params,filename)
bbo.solve()
bbo.printResult()
# bbo.isSameObject()
      
