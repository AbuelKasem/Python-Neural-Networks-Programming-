# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 01:48:40 2019

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
  #   versicolor_vs_setosa    
versicolor_desired_classification_slice=slice(0,60,1)         
versicolor_vs_setosa_desired_out=versicolor_desired_classification[versicolor_desired_classification_slice]

#print(setosa_desired_classification)
#print(versicolor_vs_setosa_desired_out)
training_set_setosa_vs_versicolor_slice=slice(0,60,1) 
versicolor_vs_setosa_train_set=training_set[training_set_setosa_vs_versicolor_slice]
                #versicolor_vs_virginica
versicolor_desired_classification_slice=slice(30,90,1)         
versicolor_vs_virginica_desired_out=versicolor_desired_classification[versicolor_desired_classification_slice]

#print(setosa_desired_classification)
#print(versicolor_vs_virginica_desired_out)
training_set_virginica_vs_versicolor_slice=slice(30,90,1) 
versicolor_vs_virginica_train_set=training_set[training_set_virginica_vs_versicolor_slice]

#   virginica_vs_setosa    

setosa_desired_classification_slice=slice(0,60,1)         
virginica_vs_setosa_desired_out=setosa_desired_classification[setosa_desired_classification_slice]
#print(setosa_desired_classification)
print(virginica_vs_setosa_desired_out)
virginica_vs_setosa_training_set=np.concatenate((virginica_training_data,setosa_training_data)) #training data set


l=0.1

e=0
n_it=100
def versicolor_vs_setosa_model_train(training_data,desired_output,iter_limit,learning_rate):
  n_err=0
  n_correct=0
  #init. error to zero ,this error is between desired output and estimated output based on decision boundary evaluation
  a=np.ones((training_data.shape[0],1)) #creating array of ones same dim. as data
  training_data=np.hstack((training_data,a)) #appending the fixed added input of 1 to the inputs
       
  iter_counter=0     
  init_weights=np.zeros(len(training_data[0])) #init. weights vector to zeros
  estimated_output=np.zeros(len(training_data)) #init. est. output vector to zeros
  
  for n in range(iter_limit): #loop through data set till stopping limit is reached
   
   for i in range(len(training_data)):  #apply perceptron algorithm on each input of training data
      
      
      iter_counter=iter_counter+1
      decision_boundary=np.dot(init_weights,training_data[i]) #scalar product of the weights vector and input vector , =sum(wi*xi)
      if( (desired_output[i]==1) and (decision_boundary <= 0)): #if the sample belongs to setosa and the decision boundary is less than or equal zero 
          #based on the learning rules ,setosa was labelled as 1 and non_setosa as -1 in the correct output array
          init_weights=init_weights+learning_rate*training_data[i] #update the weight vector by adding the sample to the weight
          
      elif( (desired_output[i]==-1) and (decision_boundary > 0)): #if the sample is non setosa and the decision boundary is greater than zero 
          init_weights=init_weights-learning_rate*training_data[i] #,then subtract input from weight 
        
      else:  #else ,if the wieghts resulted in correct classification
          estimated_output[i]=desired_output[i] #assign the desired output to the estimated output
          
          if(estimated_output[i]==desired_output[i]):
             n_correct=n_correct+1 
  final_weights=init_weights
  print(f'n_err={n_err}')
  print(f'n_correct={n_correct}')
  print(f'iter_counter={iter_counter}')
  return final_weights,estimated_output



versicolor_vs_setosa_model_weights,versicolor_vs_setosa_model_out=versicolor_vs_setosa_model_train(versicolor_vs_setosa_train_set,versicolor_vs_setosa_desired_out,n_it,l) #call the training function ,pass training data set,desired output ,number of iterations ,learning rate
                                        #store the output weigths of the model and estimated output in two variables
print(f'versicolor_vs_setosa_model_weights={versicolor_vs_setosa_model_weights}')
print(f'versicolor_vs_setosa_model_out={versicolor_vs_setosa_model_out}') 
#print(versicolor_desired_classification)
def versicolor_vs_setosa_classify(sample,model):
    sample=np.append(sample,[1])
    decision_boundary=np.dot(model,sample)
    #print(f'decision_boundary={decision_boundary}')
    if(decision_boundary> 0):
          out=1
    else:
          out=0
    return out

#s=np.array([5.0	,3.4,	1.5,	0.2])
#s2=np.array([5.0	,3.4,	1.5,	0.2])
#d=setosa_classify(s,w_f)
#print(f'd={d}')   
''''
#s2=np.array([7,3.2,4.7,1.4])
test_s=(input('enter PL,PW,SL & SW  ')) #enter 4 features of a sample input
temp = list(map(float, test_s.split())) #convert four inputs to list of four floats
test_s=np.array(temp)  #convert four inputs to an numpy array
d1=versicolor_vs_setosa_classify(test_s,versicolor_vs_setosa_model_weights) #pass data to classifier with the trained model 
print(f'd1={d1}')  


if(d1>0):
    print("sample is versicolor")
else:
    print("sample is setosa")
    
 '''   
def versicolor_vs_virginica_model_train(training_data,desired_output,iter_limit,learning_rate):
  n_err=0
  n_correct=0
   #init. error to zero ,this error is between desired output and estimated output based on decision boundary evaluation
  a=np.ones((training_data.shape[0],1)) #creating array of ones same dim. as data
  training_data=np.hstack((training_data,a)) #appending the fixed added input of 1 to the inputs
       
  iter_counter=0     
  init_weights=np.zeros(len(training_data[0])) #init. weights vector to zeros
  estimated_output=np.zeros(len(training_data)) #init. est. output vector to zeros
  
  for n in range(iter_limit): #loop through data set till stopping limit is reached
   
   for i in range(len(training_data)):  #apply perceptron algorithm on each input of training data
      
      
      iter_counter=iter_counter+1
      decision_boundary=np.dot(init_weights,training_data[i]) #scalar product of the weights vector and input vector , =sum(wi*xi)
      if( (desired_output[i]==1) and (decision_boundary <= 0)): #if the sample belongs to setosa and the decision boundary is less than or equal zero 
          #based on the learning rules ,setosa was labelled as 1 and non_setosa as -1 in the correct output array
          init_weights=init_weights+learning_rate*training_data[i] #update the weight vector by adding the sample to the weight
          
      elif( (desired_output[i]==-1) and (decision_boundary > 0)): #if the sample is non setosa and the decision boundary is greater than zero 
          init_weights=init_weights-learning_rate*training_data[i] #,then subtract input from weight 
        
      else:  #else ,if the wieghts resulted in correct classification
          estimated_output[i]=desired_output[i] #assign the desired output to the estimated output
          
          if(estimated_output[i]==desired_output[i]):
             n_correct=n_correct+1 
  final_weights=init_weights
  print(f'n_err={n_err}')
  print(f'n_correct={n_correct}')
  print(f'iter_counter={iter_counter}')
  return final_weights,estimated_output



versicolor_vs_virginica_model_weights,versicolor_vs_virginica_model_out=versicolor_vs_virginica_model_train(versicolor_vs_virginica_train_set,versicolor_vs_virginica_desired_out,n_it,l) #call the training function ,pass training data set,desired output ,number of iterations ,learning rate
                                        #store the output weigths of the model and estimated output in two variables
print(f'versicolor_vs_virginica_model_weights={versicolor_vs_virginica_model_weights}')
print(f'versicolor_vs_virginica_model_out={versicolor_vs_virginica_model_out}') 
#print(versicolor_desired_classification)
def versicolor_vs_virginica_classify(sample,model):
    sample=np.append(sample,[1])
    decision_boundary=np.dot(model,sample)
    #print(f'decision_boundary={decision_boundary}')
    if(decision_boundary> 0):
          out=1
    else:
          out=0
    return out  
'''  
test_s2=(input('enter PL,PW,SL & SW  ')) #enter 4 features of a sample input
temp = list(map(float, test_s2.split())) #convert four inputs to list of four floats
test_s2=np.array(temp)  #convert four inputs to an numpy array

d2=versicolor_vs_virginica_classify(test_s,versicolor_vs_virginica_model_weights) #pass data to classifier with the trained model 
print(f'd2={d2}')  


if(d2>0):
    print("sample is versicolor")
else:
    print("sample is virginica")
    
'''  
def virginica_vs_setosa_model_train(training_data,desired_output,iter_limit,learning_rate):
  n_err=0
  n_correct=0
    #init. error to zero ,this error is between desired output and estimated output based on decision boundary evaluation
  a=np.ones((training_data.shape[0],1)) #creating array of ones same dim. as data
  training_data=np.hstack((training_data,a)) #appending the fixed added input of 1 to the inputs
       
  iter_counter=0     
  init_weights=np.zeros(len(training_data[0])) #init. weights vector to zeros
  estimated_output=np.zeros(len(training_data)) #init. est. output vector to zeros
  
  for n in range(iter_limit): #loop through data set till stopping limit is reached
   
   for i in range(len(training_data)):  #apply perceptron algorithm on each input of training data
      
      
      iter_counter=iter_counter+1
      decision_boundary=np.dot(init_weights,training_data[i]) #scalar product of the weights vector and input vector , =sum(wi*xi)
      if( (desired_output[i]==1) and (decision_boundary <= 0)): #if the sample belongs to setosa and the decision boundary is less than or equal zero 
          #based on the learning rules ,setosa was labelled as 1 and non_setosa as -1 in the correct output array
          init_weights=init_weights+learning_rate*training_data[i] #update the weight vector by adding the sample to the weight
          
      elif( (desired_output[i]==-1) and (decision_boundary > 0)): #if the sample is non setosa and the decision boundary is greater than zero 
          init_weights=init_weights-learning_rate*training_data[i] #,then subtract input from weight 
        
      else:  #else ,if the wieghts resulted in correct classification
          estimated_output[i]=desired_output[i] #assign the desired output to the estimated output
          
          if(estimated_output[i]==desired_output[i]):
             n_correct=n_correct+1
  final_weights=init_weights
  print(f'n_err={n_err}')
  print(f'n_correct={n_correct}')
  print(f'iter_counter={iter_counter}')
  return final_weights,estimated_output



virginica_vs_setosa_model_weights,virginica_vs_setosa_model_out=virginica_vs_setosa_model_train(virginica_vs_setosa_training_set,virginica_vs_setosa_desired_out,n_it,l) #call the training function ,pass training data set,desired output ,number of iterations ,learning rate
                                        #store the output weigths of the model and estimated output in two variables
print(f'virginica_vs_setosa_model_weights={virginica_vs_setosa_model_weights}')
print(f'virginica_vs_setosa_model_out={virginica_vs_setosa_model_out}')   

def virginica_vs_setosa_classify(sample,model):
    sample=np.append(sample,[1])
    decision_boundary=np.dot(model,sample)
    #print(f'decision_boundary={decision_boundary}')
    if(decision_boundary> 0):
          out=1
    else:
          out=0
    return out
'''
#s=np.array([5.0	,3.4,	1.5,	0.2])
#s2=np.array([5.0	,3.4,	1.5,	0.2])
#d=setosa_classify(s,w_f)
#print(f'd={d}')   
'''   
def logic_model_train(training_data,desired_output,iter_limit,learning_rate):
 # n_err=0
  n_correct=0
  
  a=np.ones((training_data.shape[0],1)) #creating array of ones same dim. as data
  training_data=np.hstack((training_data,a)) #appending the fixed added input of 1 to the inputs
   
  iter_counter=0     
  init_weights=np.zeros(len(training_data[0])) #init. weights vector to zeros
  estimated_output=np.zeros(len(training_data)) #init. est. output vector to zeros
  
  
  for n in range(iter_limit): #loop through data set till stopping limit is reached
   
    for i in range(len(training_data)):  #apply perceptron algorithm on each input of training data
      
      
      iter_counter=iter_counter+1
      decision_boundary=np.dot(init_weights,training_data[i]) #scalar product of the weights vector and input vector , =sum(wi*xi)
      if( (desired_output[i]==1) and (decision_boundary <= 0)): #if the sample belongs to setosa and the decision boundary is less than or equal zero 
          #based on the learning rules ,setosa was labelled as 1 and non_setosa as -1 in the correct output array
          init_weights=init_weights+learning_rate*training_data[i] #update the weight vector by adding the sample to the weight
          
      elif( (desired_output[i]==-1) and (decision_boundary > 0)): #if the sample is non setosa and the decision boundary is greater than zero 
          init_weights=init_weights-learning_rate*training_data[i] #,then subtract input from weight 
        
      else:  #else ,if the wieghts resulted in correct classification
          estimated_output[i]=desired_output[i] #assign the desired output to the estimated output
          
          if(estimated_output[i]==desired_output[i]):
             n_correct=n_correct+1 #increment correct classification counter
   #error=desired_output-estimated_output 
    final_weights=init_weights  #assign the calcualted wieghts to final model wieghts
    
  #print(f'n_correct={n_correct}')
  #print(f'iter_counter={iter_counter}')
  return final_weights,estimated_output 

 
xx1_in=np.array([[1,1,1],[0,1,0],[1,0,0],[1,1,0],[0,1,1],[1,0,1]])
#d_out=np.array([-1,-1,-1,1])
xx1_in_classification=np.array([1,-1,1,1,-1,-1])

logic_model_weights,logic1_model_out=logic_model_train(xx1_in,xx1_in_classification,n_it,l)

print(f'logic_model_weights={logic_model_weights}')

def logic_classify(sample,model):
    sample=np.append(sample,[1])
    decision_boundary=np.dot(model,sample)
    #print(f'decision_boundary={decision_boundary}')
    if(decision_boundary> 0):
          out=1
    else:
          out=0
    return out

s2=np.array([7,3.2,4.7,1.4])
test_s=(input('enter PL,PW,SL & SW  ')) #enter 4 features of a sample input
temp = list(map(float, test_s.split())) #convert four inputs to list of four floats
test_s=np.array(temp)  #convert four inputs to an numpy array
d1=versicolor_vs_setosa_classify(test_s,versicolor_vs_setosa_model_weights) #pass data to classifier with the trained model 
print(f'd1={d1}') 

d2=versicolor_vs_virginica_classify(test_s,versicolor_vs_virginica_model_weights) #pass data to classifier with the trained model 
print(f'd2={d2}')  
d3=virginica_vs_setosa_classify(test_s,virginica_vs_setosa_model_weights) #pass data to classifier with the trained model 
print(f'd3={d3}')  

test=np.array([d1,d2,d3])
logic1_out=logic_classify(test,logic_model_weights)
print(f'logic1_out={logic1_out}')
if(logic1_out>0):
    print("sample is versicolor")
else:
    print("sample is non versicolor") 
#print(f'logic1_out={logic1_out}')

#logic_in=np.array([1,1,1])
#logic1_out=logic_classify(logic_in,logic_model_weights)

'''

#s2=np.array([7,3.2,4.7,1.4])
#test_s3=(input('enter PL,PW,SL & SW  ')) #enter 4 features of a sample input
#temp3 = list(map(float, test_s.split())) #convert four inputs to list of four floats
#test_s=np.array(temp3)  #convert four inputs to an numpy array
d3=virginica_vs_setosa_classify(test_s,virginica_vs_setosa_model_weights) #pass data to classifier with the trained model 
print(f'd3={d3}')  


if(d3>0):
    print("sample is virginica")
else:
    print("sample is setosa") 
'''