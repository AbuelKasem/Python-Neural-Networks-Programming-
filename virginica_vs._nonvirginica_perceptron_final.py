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
#setosa trian data
setosa_data_training_slice=slice(0,30,1) #the first 30 sample of setosa
setosa_training_data=setosa_data[setosa_data_training_slice]
setosa_training_data=np.array(setosa_training_data)
#setosa test data
setosa_data_test_slice=slice(30,50,1) #the remaining 20 sample
setosa_test_data=setosa_data[setosa_data_test_slice]
setosa_test_data=np.array(setosa_test_data)

#versicolor data
versicolor_iris_slice=slice(50,100,1)
versicolor_data=iris_data_matrix[versicolor_iris_slice]
#training data
versicolor_data_slice=slice(0,30,1)
versicolor_training_data=versicolor_data[versicolor_data_slice]
versicolor_training_data=np.array(versicolor_training_data)
#versicolor test data
versicolor_data_test_slice=slice(30,50,1) #the remaining 20 sample
versicolor_test_data=versicolor_data[versicolor_data_test_slice]
versicolor_test_data=np.array(versicolor_test_data)

#virginica_data

virginica_iris_slice=slice(100,150,1)
virginica_data=iris_data_matrix[virginica_iris_slice]
#training data
virginica_data_slice=slice(0,30,1)
virginica_training_data=virginica_data[virginica_data_slice]
virginica_training_data=np.array(virginica_training_data)
#virginica test data
virginica_data_test_slice=slice(30,50,1) #the remaining 20 sample
virginica_test_data=virginica_data[virginica_data_test_slice]
virginica_test_data=np.array(virginica_test_data)
#the 60 training samples
test_set=np.concatenate((setosa_test_data,versicolor_test_data,virginica_test_data)) 

#the 90 training samples
training_set=np.concatenate((setosa_training_data,versicolor_training_data,virginica_training_data)) #training data set
setosa_desired_classification=np.zeros(len(training_set))
#desired output settings ,1 if its from the class ,-1 if not
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
                   #test desired set
setosa_desired_test_classification=np.zeros(len(test_set))                 
for i in range(len(setosa_desired_test_classification)):
     if(i<=19):
         setosa_desired_test_classification[i]=setosa_desired_test_classification[i]+1
     else:
         setosa_desired_test_classification[i]=setosa_desired_test_classification[i]


versicolor_desired_test_classification=np.zeros(len(test_set))
for i in range(len(versicolor_desired_test_classification)):
     if(i>=20 and i<40):
         versicolor_desired_test_classification[i]=versicolor_desired_test_classification[i]+1
     else:
         versicolor_desired_test_classification[i]=versicolor_desired_test_classification[i]


virginica_desired_test_classification=np.zeros(len(test_set))
for i in range(len(virginica_desired_test_classification)):
     if( i>=40):
         virginica_desired_test_classification[i]=virginica_desired_test_classification[i]+1
     else:
         virginica_desired_test_classification[i]=virginica_desired_test_classification[i]         

def accuracy_evaluation(estimation,desired):   
    N_correct=0
    N_errors=0
    accuracy=0   
    for y in range (len(desired)):
        if(estimation[y]==desired[y]):
             N_correct=N_correct+1
   
        elif(estimation[y]!=desired[y]):
           N_errors=N_errors+1

    accuracy=(N_correct)/(len(desired)) *100
    return N_correct,N_errors,accuracy

l=.1
e=0
n_it=100
def virginica_model_train(training_data,desired_output,iter_limit,learning_rate):
  
  
  a=np.ones((training_data.shape[0],1)) #creating array of ones same dim. as data
  training_data=np.hstack((training_data,a)) #appending the fixed added input of 1 to the inputs
  error_n=1    
  iter_counter=0     
  init_weights=np.zeros(len(training_data[0])) #init. weights vector to zeros
  estimated_output=np.zeros(len(training_data)) #init. est. output vector to zeros
  #while(error_n!=0):
  for n in range(iter_limit): #loop through data set till stopping limit is reached
     iter_counter=iter_counter+1             
     for i in range(len(training_data)):  #apply perceptron algorithm on each input of training data
      
      
       
       decision_boundary=np.dot(init_weights,training_data[i]) #scalar product of the weights vector and input vector , =sum(wi*xi)
       if( (desired_output[i]==1) and (decision_boundary < 0)): #if the sample belongs to setosa and the decision boundary is less than or equal zero 
          #based on the learning rules ,virginica was labelled as 1 and non_virginica as -1 in the correct output array
          init_weights=init_weights+learning_rate*training_data[i] #update the weight vector by adding the sample to the weight
          
       elif( (desired_output[i]==-1) and (decision_boundary >= 0)): #if the sample is non virginica and the decision boundary is greater than zero 
          init_weights=init_weights-learning_rate*training_data[i] #,then subtract input from weight 
       
       else:  #else ,if the wieghts resulted in correct classification
          estimated_output[i]=desired_output[i] #assign the desired output to the estimated output
          
     correct_n,error_n,accuracy=accuracy_evaluation(estimated_output,desired_output)
     final_weights=init_weights  #assign the calcualted wieghts to final model wieghts
     #if(iter_counter >=100): #error is the number of misclassifications
       # break
 
  print(f'iter_counter={iter_counter}')
  
  
  final_weights=init_weights
  
  return final_weights,estimated_output


#print(virginica_desired_classification)

virginica_model_weights,virginica_model_out=virginica_model_train(training_set,virginica_desired_classification,n_it,l) #call the training function ,pass training data set,desired output ,number of iterations ,learning rate
                                        #store the output weigths of the model and estimated output in two variables
print(f'virginica_model_weights={virginica_model_weights}')
#print(f'virginica_model_out={virginica_model_out}') 
def virginica_classify(sample,model):
    sample=np.append(sample,[1])
    decision_boundary=np.dot(model,sample)
    #print(f'decision_boundary={decision_boundary}')
    if(decision_boundary> 0):
          out=1
    else:
          out=0  #the function is configured to return 0 if the sample is not in the class ,to simplify using the model in the network
    return out

  
'''
#s2=np.array([7,3.2,4.7,1.4])
test_s=(input('enter PL,PW,SL & SW  ')) #enter 4 features of a sample input
temp = list(map(float, test_s.split())) #convert four inputs to list of four floats
test_s=np.array(temp)  #convert four inputs to an numpy array
d2=virginica_classify(test_s,virginica_model_weights) #pass data to classifier with the trained model 
print(f'd2={d2}')  


if(d2>0):
    print("sample is virginica")
else:
    print("sample is non-virginica")
   
'''
 #10 missclassifications  ,20 correct   
classification_test_result=np.zeros(len(test_set))
classification_test_result1=np.zeros(len(test_set))
print('classification_test_result before')    
print(classification_test_result)  
   
for x in range (len(test_set)):
    
   classification_test_result1[x]=  virginica_classify(test_set[x],virginica_model_weights)
   

print('classification_test_result after')    
print(classification_test_result1)       
    




crct,err,acc=accuracy_evaluation(classification_test_result1,virginica_desired_test_classification)
print(f'err={err}')
print(f'crct={crct}')
print(f'acc={acc}')
print('virginica_desired_test_classification')    
print(virginica_desired_test_classification)


