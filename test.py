# -*- coding:utf-8 -*-
import numpy as np
import random


rnd = np.random.random(5)
# print rnd 

#print np.zeros(5)
#print np.zeros((3,2))
alist = [123,'xyz','zara','abc','xyz']
print alist
alist.sort() # 修改list本身
print alist

blist = [4,5,1,5,2,0,9,6]
print blist
print sorted(blist) # sorted返回一个新的list
print np.zeros(5)
print range(1,11/2 + 1)
print 2!=2

print range(10-1,0,-1)

bound = np.tile([[-3],[3]],5)
print 0.02*(bound[1] - bound[0])
print int(round(0.3 * 34))
a = [1,2,3]
b = [4,5,6,7,8]
a.extend(b)
print a