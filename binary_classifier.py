# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:17:40 2019

@author: Kasem
"""

import numpy as np
import numpy.matlib 
from matplotlib import pyplot as plt

#x1=np.array([[1,2,1],[3,4,1]])
#x2=np.array([[6,7,1],[8,9,1]])
#x=np.concatenate((x1, x2))
out=np.array([0,0,0])

#x=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
x=np.array([[1,0,1],[1,1,1],[0,1,0]])
#d_out=np.array([-1,-1,-1,1])
d_out=np.array([1,-1,-1])
l=1
w=np.array([0,0,0])
#for i in range(len(x)):
   #print(f'x{i} ={x[i]}')
e=0
n_it=10

#def model_train(training_data,desired_output,init_weights,iter_limit,learning_rate):
def model_train(training_data,desired_output,iter_limit,learning_rate):
  n_err=0
  n_correct=0
  error=0
  a=np.ones((training_data.shape[0],1))
  training_data=np.hstack((training_data,a))
       
       
  init_weights=np.zeros(len(training_data[0]))
  estimated_output=np.zeros(len(training_data))
  
  for n in range(iter_limit): 
   
   for i in range(len(training_data)):
      
      
    
      decision_boundary=np.dot(init_weights,training_data[i])
      if(decision_boundary> 0):
          
          estimated_output[i]=1
          
      else:
          
         estimated_output[i]=-1
         
      if(estimated_output[i]!=desired_output[i]):
          
         error=desired_output[i]-estimated_output[i]
         init_weights=init_weights+learning_rate*(error)*training_data[i]
         n_err=n_err+1
      else:
         n_correct=n_correct+1
  final_weights=init_weights
  print(n_err)
  print(n_correct)
  return final_weights,estimated_output



w_f,e_out=model_train(x,d_out,n_it,l)
model=w_f
  
        
#print(c)     
print(f'w_f={w_f}')
print(f'e_out={e_out}') 

def classify(sample,model):
    sample=np.append(sample,[1])
    decision_boundary=np.dot(model,sample)
    if(decision_boundary> 0):
          out=1
    else:
          out=-1
    return out

s=np.array([0,1,0])

d=classify(s,w_f)
print(f'd={d}')
          
  
     