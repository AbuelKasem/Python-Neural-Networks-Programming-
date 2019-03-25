# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:04:50 2019

@author: Kasem
"""

from numpy import *
import numpy as np
from matplotlib import pyplot as plt

data=loadtxt('iris.dat',unpack=True)
#print(data)

iris_data_matrix = np.genfromtxt("iris.dat")  #generate array from text file matching the same format of file
setosa_iris_slice=slice(0,50,1)
setosa_data=iris_data_matrix[setosa_iris_slice] 
setosa_data_slice=slice(0,30,1)
setosa_training_data=setosa_data[setosa_data_slice]
setosa_training_data=np.array(setosa_training_data)
print('setosa_data')
print(setosa_data)
print(setosa_data[0])
print('setosa_training_data')
print(setosa_training_data)

versicolor_iris_slice=slice(50,100,1)
versicolor_data=iris_data_matrix[versicolor_iris_slice]
versicolor_data_slice=slice(0,30,1)
versicolor_training_data=versicolor_data[versicolor_data_slice]
versicolor_training_data=np.array(versicolor_training_data)
print('versicolor_data')
print(versicolor_data)
print(versicolor_data[0])
print('versicolor_training_data')
print(versicolor_training_data)

virginica_iris_slice=slice(100,150,1)
virginica_data=iris_data_matrix[virginica_iris_slice]
virginica_data_slice=slice(0,30,1)
virginica_training_data=virginica_data[virginica_data_slice]
virginica_training_data=np.array(virginica_training_data)
print('virginica_data')
print(virginica_data)
print(virginica_data[0])
print('virginica_training_data')
print(virginica_training_data)
print('training_data')
training_set=np.concatenate((setosa_training_data,versicolor_training_data,virginica_training_data))
#training_set=np.append(setosa_training_data,[versicolor_training_data],[virginica_training_data])
print(training_set)
setosa_desired_classification=np.zeros(len(training_set))
print(setosa_desired_classification)
#c=-1*(setosa_desired_classification[31:90])
#print(c)
for i in range(len(setosa_desired_classification)):
     if(i<=29):
         setosa_desired_classification[i]=setosa_desired_classification[i]+1
     else:
         setosa_desired_classification[i]=setosa_desired_classification[i]-1
print('setosa_desired_classification')         
print(setosa_desired_classification)

versicolor_desired_classification=np.zeros(len(training_set))
for i in range(len(versicolor_desired_classification)):
     if(i>=30 and i<60):
         versicolor_desired_classification[i]=versicolor_desired_classification[i]+1
     else:
         versicolor_desired_classification[i]=versicolor_desired_classification[i]-1
print('versicolor_desired_classification')           
print(versicolor_desired_classification)

virginica_desired_classification=np.zeros(len(training_set))
for i in range(len(virginica_desired_classification)):
     if( i>=60):
         virginica_desired_classification[i]=virginica_desired_classification[i]+1
     else:
         virginica_desired_classification[i]=virginica_desired_classification[i]-1
print('virginica_desired_classification')           
print(virginica_desired_classification)


m=np.array([1,2,3,4,5,6])
#n=m[3:2:]
n=np.delete(m,2,axis = 0)
#print(m)
print(n)
a = np.array([1, 2, 5, 6, 7])

ind2remove = [1: 3]

print (np.delete(a, ind2remove))
print( a)

