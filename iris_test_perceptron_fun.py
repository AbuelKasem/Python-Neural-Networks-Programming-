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
x=np.array([[5.1,3.5,1.4,0.2],[4.9,3,1.4,0.2],[4.7,3.2,1.3,0.2],[5.8,2.7,5.1,1.9],[6.4,3.2,3.5,1.5],[7,3.2,4.7,1.4]])
#d_out=np.array([-1,-1,-1,1])
d_out=np.array([1,1,1,-1,-1,-1])
l=1
w=np.array([0,0,0,0,0])
#for i in range(len(x)):
   #print(f'x{i} ={x[i]}')
e=0
n_it=10
miss=0
def model_train(training_data,desired_output,iter_limit,learning_rate):
  n_err=0
  n_correct=0
  error=0  #init. error to zero ,this error is between desired output and estimated output based on decision boundary evaluation
  a=np.ones((training_data.shape[0],1)) #creating array of ones same dim. as data
  training_data=np.hstack((training_data,a)) #appending the fixed added input of 1 to the inputs
       
       
  init_weights=np.zeros(len(training_data[0])) #init. weights vector to zeros
  estimated_output=np.zeros(len(training_data)) #init. est. output vector to zeros
  
  for n in range(iter_limit): #loop through data set till stopping limit is reached
   
   for i in range(len(training_data)):  #apply perceptron algorithm on each input of training data
      
      
    
      decision_boundary=np.dot(init_weights,training_data[i]) #scalar product of the weights vector and input vector , =sum(wi*xi)
      if(decision_boundary> 0): 
          estimated_output[i]=1
      else:
         estimated_output[i]=-1
      if(estimated_output[i]!=desired_output[i]):
          
         error=desired_output[i]-estimated_output[i]
         init_weights=init_weights+learning_rate*(error)*training_data[i]
         n_err=n_err+1
      else:
         n_correct=n_correct+1 #wieghts update equation
  final_weights=init_weights
  
  return final_weights,estimated_output



w_f,e_out=model_train(x,d_out,n_it,l)
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

s=np.array([5.0	,3.4,	1.5,	0.2])
s2=np.array([5.0	,3.4,	1.5,	0.2])
d=classify(s,w_f)
print(f'd={d}')   

#s2=np.array([7,3.2,4.7,1.4])
test_s=(input('enter PL,PW,SL & SW  '))
temp = list(map(float, test_s.split()))
test_s=np.array(temp)
d2=classify(test_s,w_f)
print(f'd2={d2}')  


if(d2>0):
    print("sample is setosa")
else:
    print("sample is non-setosa")
     

    
     