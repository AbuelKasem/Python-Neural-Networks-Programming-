# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:59:37 2019

@author: Kasem
"""
import numpy as np
import numpy.matlib 
from matplotlib import pyplot as plt

n_iteration=100
error=0
in_data=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
print(in_data)
desired_out=np.array([0,0,0,1])
#weights=np.zeros(3)
weights=np.array([0,0,0])


def classify(weights,in_data):
    decision_boundary=np.dot(weights,in_data)
    out_decision=0
    if (decision_boundary>0):
        out_decision=1;
    else:
        out_decision=0;
    return out_decision


def w_update(weights,input_data,d_output):
    for i in range(n_iteration):
        #zip(range(x), range(y)):
        for x,y in zip(input_data,d_output):
            estimation=classify(weights,input_data)
              weights=weights+error*input_data
              
              error=d_output-estimation
    return weights


              
w_update(weights,in_data,desired_out)   
b=np.array([1, 1])
classify(weights,b)
        