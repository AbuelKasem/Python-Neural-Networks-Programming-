# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:46:12 2019

@author: Kasem
"""
import numpy as np
import numpy.matlib 
from matplotlib import pyplot as plt

#x1=np.array([[1,2,1],[3,4,1]])
#x2=np.array([[6,7,1],[8,9,1]])
#x=np.concatenate((x1, x2))
out=np.array([0,0,0,0])

x=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
#x=np.array([[1,0,1],[1,1,1]])
d_out=np.array([-1,-1,-1,1])
#d_out=np.array([-1,1])
l=1
w=np.array([0,0,0])
#for i in range(len(x)):
   #print(f'x{i} ={x[i]}')
e=0
n_it=50
ier=1; 

for i in range(len(x)):
    for n in range(n_it):
      print(f'x{i} ={x[i]}') 
    
      d_b=np.dot(w,x[i])
      if(d_b < 0 and d_out[i]==1): 
         w=w+l*x[i]
         e=e+1 
         out[i]=d_b
     # else:
          #out[i]=d_out[i]
       
      if(d_b >= 0 and d_out[i]==-1): 
          w=w-l*x[i]
          e=e+1 
          out[i]=d_b
      #else:
          #out[i]=d_out[i]
      err=d_out[i]-d_b
      print(f'w={w}')
    
 
  
        
      
print(f'w={w}')
print(f'out={out}') 
    
     