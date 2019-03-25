# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:13:48 2019

@author: Kasem
"""

import numpy as np
import numpy.matlib 
from matplotlib import pyplot as plt
# x1,x2 of c1   sum(w*x) > 0
# x4,x4 of c2   sum(w*x) < 0

x1=np.array([1,2,1])
x2=np.array([3,4,1])
x3=np.array([6,7,1])
x4=np.array([8,9,1])
w=np.array([0,0,0])
print(x1)
print(x2)
print(x3)
print(x4)
print(w)
print(x1*x2)
a = np.array([1,2]) 
b = np.array([3,4]) 
wxsum=np.dot(a,b)   #dot product
print(wxsum)
decision_boundary=np.dot(w,x1)
print(f'decision_boundary={decision_boundary}')
if (decision_boundary>0): 
    print('x1 of c1')
else: 
    w=w+x1
    print(f'w={w}')
   
decision_boundary=np.dot(w,x1)
print(f'decision_boundary={decision_boundary}')
if (decision_boundary>0): 
    print('x1 of c1')
    print(f'w={w}')
else: 
    w=w+x1
    print(f'w={w}')
decision_boundary=np.dot(w,x2)
print(f'decision_boundary={decision_boundary}')
if (decision_boundary>0): 
    print('x2 of c1')
    print(f'w={w}')
else: 
    w=w+x2
    print(f'w={w}')
   
decision_boundary=np.dot(w,x3)
print(f'decision_boundary={decision_boundary}')
if (decision_boundary>0): 
    print('x1 of c1')
    print(f'w={w}')
else: 
    w=w+x2
    print(f'w={w}')

   
