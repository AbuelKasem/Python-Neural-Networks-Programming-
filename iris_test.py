# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 03:23:22 2019

@author: Kasem
"""

import numpy as np
import numpy.matlib 
from matplotlib import pyplot as plt

#x1=np.array([[1,2,1],[3,4,1]])
#x2=np.array([[6,7,1],[8,9,1]])
#x=np.concatenate((x1, x2))
out=np.array([0,0,0,0,0,0])

#x=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
x=np.array([[5.1,3.5,1.4,0.2,1],[4.9,3,1.4,0.2,1],[4.7,3.2,1.3,0.2,1],[5.8,2.7,5.1,1.9,1],[6.4,3.2,3.5,1.5,1],[7,3.2,4.7,1.4,1]])
#d_out=np.array([-1,-1,-1,1])
desired_out=np.array([1,1,1,-1,-1,-1])
l=0.1
w=np.array([0,0,0,0,0])
#for i in range(len(x)):
   #print(f'x{i} ={x[i]}')
e=0
n_iteration=13
miss=0

for n in range(n_iteration):
   c=0
   
   print(n)
   for i in range(len(x)):
      
      print(f'x{i} ={x[i]}') 
    
      decision_boundary=np.dot(w,x[i])
      if(decision_boundary > 0):
          out[i]=1
      else:
         out[i]=-1
      if(out[i]==desired_out[i]):
          c=c+1
      else:
         error=desired_out[i]-out[i]
         w=w+l*(error)*x[i]
         miss=miss+1
      print(f'w={w}')
    
 
  
print(miss)        
print(c)     
print(f'w={w}')
print(f'out={out}') 


