import random

def randomSelectThreeIndices(sizepop,index):
    r1 = random.randint(0,sizepop-1)
    while r1 == index:
        r1 = random.randint(0,sizepop-1)
    r2 = random.randint(0,sizepop-1)
    while r2 == index or r2 == r1:
        r2 = random.randint(0,sizepop-1)
    r3 = random.randint(0,sizepop-1)
    while r3 == index or r3 == r1 or r3 == r2:
        r3 = random.randint(0,sizepop-1)
    return r1,r2,r3

# print randomSelectThreeIndices(10, 4)