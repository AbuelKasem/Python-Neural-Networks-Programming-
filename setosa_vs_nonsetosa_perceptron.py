# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 03:23:22 2019

@author: Kasem
"""

import numpy as np
import numpy.matlib 
from matplotlib import pyplot as plt


iris_data_matrix = np.genfromtxt("iris.dat")  #generate array from text file matching the same format of file
setosa_iris_slice=slice(0,50,1)
setosa_data=iris_data_matrix[setosa_iris_slice] 
setosa_data_slice=slice(0,30,1)
setosa_training_data=setosa_data[setosa_data_slice]
setosa_training_data=np.array(setosa_training_data)
versicolor_iris_slice=slice(50,100,1)
versicolor_data=iris_data_matrix[versicolor_iris_slice]
versicolor_data_slice=slice(0,30,1)
versicolor_training_data=versicolor_data[versicolor_data_slice]
versicolor_training_data=np.array(versicolor_training_data)
virginica_iris_slice=slice(100,150,1)
virginica_data=iris_data_matrix[virginica_iris_slice]
virginica_data_slice=slice(0,30,1)
virginica_training_data=virginica_data[virginica_data_slice]
virginica_training_data=np.array(virginica_training_data)
training_set=np.concatenate((setosa_training_data,versicolor_training_data,virginica_training_data)) #training data set
setosa_desired_classification=np.zeros(len(training_set))
for i in range(len(setosa_desired_classification)):
     if(i<=29):
         setosa_desired_classification[i]=setosa_desired_classification[i]+1
     else:
         setosa_desired_classification[i]=setosa_desired_classification[i]-1


versicolor_desired_classification=np.zeros(len(training_set))
for i in range(len(versicolor_desired_classification)):
     if(i>=30 and i<60):
         versicolor_desired_classification[i]=versicolor_desired_classification[i]+1
     else:
         versicolor_desired_classification[i]=versicolor_desired_classification[i]-1


virginica_desired_classification=np.zeros(len(training_set))
for i in range(len(virginica_desired_classification)):
     if( i>=60):
         virginica_desired_classification[i]=virginica_desired_classification[i]+1
     else:
         virginica_desired_classification[i]=virginica_desired_classification[i]-1

#x=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
x=np.array([[5.1,3.5,1.4,0.2],[4.9,3,1.4,0.2],[4.7,3.2,1.3,0.2],[5.8,2.7,5.1,1.9],[6.4,3.2,3.5,1.5],[7,3.2,4.7,1.4]])
#d_out=np.array([-1,-1,-1,1])
d_out=np.array([1,1,1,-1,-1,-1])
l=1

e=0
n_it=4
def setosa_model_train(training_data,desired_output,iter_limit,learning_rate):
  n_err=0
  n_correct=0
  error=0  #init. error to zero ,this error is between desired output and estimated output based on decision boundary evaluation
  a=np.ones((training_data.shape[0],1)) #creating array of ones same dim. as data
  training_data=np.hstack((training_data,a)) #appending the fixed added input of 1 to the inputs
       
  iter_counter=0     
  init_weights=np.zeros(len(training_data[0])) #init. weights vector to zeros
  estimated_output=np.zeros(len(training_data)) #init. est. output vector to zeros
  
  for n in range(iter_limit): #loop through data set till stopping limit is reached
   
   for i in range(len(training_data)):  #apply perceptron algorithm on each input of training data
      
      
      iter_counter=iter_counter+1
      decision_boundary=np.dot(init_weights,training_data[i]) #scalar product of the weights vector and input vector , =sum(wi*xi)
      if(decision_boundary> 0): 
          estimated_output[i]=1
      else:
         estimated_output[i]=-1
      if(estimated_output[i]!=desired_output[i]):
          
         error=desired_output[i]-estimated_output[i]
         init_weights=init_weights+learning_rate*(error)*training_data[i]#wieghts update equation
         n_err=n_err+1
      else:
         n_correct=n_correct+1 
  final_weights=init_weights
  print(f'n_err={n_err}')
  print(f'n_correct={n_correct}')
  print(f'iter_counter={iter_counter}')
  return final_weights,estimated_output



setosa_model_weights,setosa_model_out=setosa_model_train(training_set,setosa_desired_classification,n_it,l) #call the training function ,pass training data set,desired output ,number of iterations ,learning rate
                                        #store the output weigths of the model and estimated output in two variables
print(f'setosa_model_weights={setosa_model_weights}')
print(f'setosa_model_out={setosa_model_out}') 
print(versicolor_desired_classification)
def setosa_classify(sample,model):
    sample=np.append(sample,[1])
    decision_boundary=np.dot(model,sample)
    #print(f'decision_boundary={decision_boundary}')
    if(decision_boundary> 0):
          out=1
    else:
          out=-1
    return out

#s=np.array([5.0	,3.4,	1.5,	0.2])
#s2=np.array([5.0	,3.4,	1.5,	0.2])
#d=setosa_classify(s,w_f)
#print(f'd={d}')   

#s2=np.array([7,3.2,4.7,1.4])
test_s=(input('enter PL,PW,SL & SW  ')) #enter 4 features of a sample input
temp = list(map(float, test_s.split())) #convert four inputs to list of four floats
test_s=np.array(temp)  #convert four inputs to an numpy array
d2=setosa_classify(test_s,setosa_model_weights) #pass data to classifier with the trained model 
print(f'd2={d2}')  


if(d2>0):
    print("sample is setosa")
else:
    print("sample is non-setosa")
   

    
     