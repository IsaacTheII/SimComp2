#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Name:      Simon Padua
   Email:     simon.padua@uzh.ch
   Date:      01/04/2020
   Course:    
   Semester:  
   Week:      
   Thema:     
"""

import numpy as np
from copy import copy
from time import time


class Test:
    def __init__(self, par1, par2, ref=None):
        self.field1 = par1
        self.field2 = par2
        self.reference = ref


r = Test(8, 9)
test = Test(1, 2, r)
t2 = copy(test)
t2.field1 = 3
t2.reference.field1 = 7
print(test.field1, t2.field1)
print()

a = np.array([1,2,3])
b = np.array([6,7,8])
list1 = [np.array([6,7,8]) for _ in range(1000000)]
list2 = [np.array([6,7,8]) for _ in range(1000000)]
list3 = [np.array([6,7,8]) for _ in range(1000000)]


t = time()
for i in range(1000000):
    np.dot(a, b)
print(time() - t)

t = time()
for i in range(1000000):
    a.dot(b)
print(time() - t)


t = time()
for i in range(10000000):
    a-b
print(time() - t)

t = time()
for i in range(10000000):
    np.subtract(a, b)
print(time() - t)



t = time()
for i in range(1000000):
    list1[i] = np.zeros(3)
print(time() - t)

t = time()
for i in range(1000000):
    list2[i] = np.zeros_like(list2[i])
print(time() - t)

t = time()
for i in range(1000000):
    list3[i] * 0
print(time() - t)


t = time()
for _ in range(100000):
    np.linalg.norm(a-b)
print(time() - t)

t = time()
for _ in range(100000):
    d = a-b
    np.sqrt(d.dot(d))
print(time() - t)


t = time()
for _ in range(100000000):
    r.field1 += 1
    test.field1 += 1
print(time() - t)



t = time()
for _ in range(100000000):
    r.field1 += 1
for _ in range(100000000):
    test.field1 += 1
print(time() - t)
