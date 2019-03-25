# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 03:50:35 2019

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

l=0.1 #learning rate


n_it=100 #stopping criterion , max number of iterations




def setosa_model_train(training_data,desired_output,iter_limit,learning_rate):
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
  return final_weights,estimated_output  #return final wiegths and calculated classification



setosa_model_weights,setosa_model_out=setosa_model_train(training_set,setosa_desired_classification,n_it,l) #call the training function ,pass training data set,desired output ,number of iterations ,learning rate
                                        #store the output weigths of the model and estimated output in two variables
#print(f'setosa_model_weights={setosa_model_weights}') #print the model
#print(f'setosa_model_out={setosa_model_out}') 
#print(setosa_desired_classification)
def setosa_classify(sample,model):   #classifier function ,takes input the sample and the caluclated weights model from trainer
    sample=np.append(sample,[1])  #add 1 as a fixed 5th input 
    decision_boundary=np.dot(model,sample)  #apply the decision boundary to the sample
    #print(f'decision_boundary={decision_boundary}')
    if(decision_boundary> 0):  #test the sample against classification criteria
          out=1   #if it is greater than 0 ,the sample belongs to setosa class
    else:
          out=0  #else if the decision boundary is less than zero ,it is not setosa
    return out




def versicolor_model_train(training_data,desired_output,iter_limit,learning_rate):
  n_err=0
  n_correct=0
  #error=0  #init. error to zero ,this error is between desired output and estimated output based on decision boundary evaluation
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
          #based on the learning rules ,virginica was labelled as 1 and non_virginica as -1 in the correct output array
          init_weights=init_weights+learning_rate*training_data[i] #update the weight vector by adding the sample to the weight
          n_err=n_err+1
      elif( (desired_output[i]==-1) and (decision_boundary > 0)): #if the sample is non virginica and the decision boundary is greater than zero 
          init_weights=init_weights-learning_rate*training_data[i] #,then subtract input from weight 
          n_err=n_err+1
      else:  #else ,if the wieghts resulted in correct classification
          estimated_output[i]=desired_output[i] #assign the desired output to the estimated output
          
          if(estimated_output[i]==desired_output[i]):
             n_correct=n_correct+1 #increment correct classification counter1 
  final_weights=init_weights
  #print(f'n_err={n_err}')
  #print(f'n_correct={n_correct}')
  #print(f'iter_counter={iter_counter}')
  return final_weights,estimated_output



versicolor_model_weights,versicolor_model_out=versicolor_model_train(training_set,versicolor_desired_classification,n_it,l) #call the training function ,pass training data set,desired output ,number of iterations ,learning rate
                                        #store the output weigths of the model and estimated output in two variables
#print(f'versicolor_model_weights={versicolor_model_weights}')
#print(f'versicolor_model_out={versicolor_model_out}') 

def versicolor_classify(sample,model):
    sample=np.append(sample,[1])
    decision_boundary=np.dot(model,sample)
    if(decision_boundary> 0):
          out=1
    else:
          out=0
    return out


def virginica_model_train(training_data,desired_output,iter_limit,learning_rate):
  n_err=0
  n_correct=0
  #error=0  #init. error to zero ,this error is between desired output and estimated output based on decision boundary evaluation
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
          #based on the learning rules ,virginica was labelled as 1 and non_virginica as -1 in the correct output array
          init_weights=init_weights+learning_rate*training_data[i] #update the weight vector by adding the sample to the weight
          n_err=n_err+1
      elif( (desired_output[i]==-1) and (decision_boundary > 0)): #if the sample is non virginica and the decision boundary is greater than zero 
          init_weights=init_weights-learning_rate*training_data[i] #,then subtract input from weight 
          n_err=n_err+1
      else:  #else ,if the wieghts resulted in correct classification
          estimated_output[i]=desired_output[i] #assign the desired output to the estimated output
          
          if(estimated_output[i]==desired_output[i]):
             n_correct=n_correct+1 #increment correct classification counter
  final_weights=init_weights
 
  return final_weights,estimated_output




virginica_model_weights,virginica_model_out=virginica_model_train(training_set,virginica_desired_classification,n_it,l) #call the training function ,pass training data set,desired output ,number of iterations ,learning rate
                                        #store the output weigths of the model and estimated output in two variables
#print(f'virginica_model_weights={virginica_model_weights}')
#print(f'virginica_model_out={virginica_model_out}') 
def virginica_classify(sample,model):
    sample=np.append(sample,[1])
    decision_boundary=np.dot(model,sample)
    #print(f'decision_boundary={decision_boundary}')
    if(decision_boundary> 0):
          out=1
    else:
          out=0
    return out



def logic1_model_train(training_data,desired_output,iter_limit,learning_rate):
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

 
xx1_in=np.array([[0,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,1],[1,0,1]])
#d_out=np.array([-1,-1,-1,1])
xx1_in_classification=np.array([-1,-1,-1,1,-1,-1])

logic1_model_weights,logic1_model_out=logic1_model_train(xx1_in,xx1_in_classification,n_it,l)

#print(f'logic1_model_weights={logic1_model_weights}')

def logic1_classify(sample,model):
    sample=np.append(sample,[1])
    decision_boundary=np.dot(model,sample)
    #print(f'decision_boundary={decision_boundary}')
    if(decision_boundary> 0):
          out=1
    else:
          out=0
    return out



logic1_in=np.array([1,1,1])
logic1_out=logic1_classify(logic1_in,logic1_model_weights)
#print(f'logic1_out={logic1_out}')

def logic2_model_train(training_data,desired_output,iter_limit,learning_rate):
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

yy1_in=np.array([[0,0,0],[1,0,0],[0,0,1],[0,1,0],[0,1,1],[1,1,0]])
#d_out=np.array([-1,-1,-1,1])
yy1_in_classification=np.array([-1,-1,-1,1,-1,-1])

logic2_model_weights,logic2_model_out=logic2_model_train(yy1_in,yy1_in_classification,n_it,l)

#print(f'logic2_model_weights={logic2_model_weights}')

def logic2_classify(sample,model):
    sample=np.append(sample,[1])
    decision_boundary=np.dot(model,sample)
    #print(f'decision_boundary={decision_boundary}')
    if(decision_boundary> 0):
          out=1
    else:
          out=0
    return out



logic2_in=np.array([0,1,0])
logic2_out=logic2_classify(logic2_in,logic2_model_weights)
#print(f'logic2_out={logic2_out}')



def logic3_model_train(training_data,desired_output,iter_limit,learning_rate):
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

zz1_in=np.array([[0,0,0],[1,0,1],[0,0,1],[0,0,1],[0,1,1],[1,1,0]])
#d_out=np.array([-1,-1,-1,1])
zz1_in_classification=np.array([-1,-1,-1,1,-1,-1])

logic3_model_weights,logic3_model_out=logic3_model_train(zz1_in,zz1_in_classification,n_it,l)

#print(f'logic3_model_weights={logic3_model_weights}')

def logic3_classify(sample,model):
    sample=np.append(sample,[1])
    decision_boundary=np.dot(model,sample)
    #print(f'decision_boundary={decision_boundary}')
    if(decision_boundary> 0):
          out=1
    else:
          out=0
    return out



logic3_in=np.array([0,1,1])
logic3_out=logic2_classify(logic3_in,logic3_model_weights)
#print(f'logic3_out={logic3_out}')


test_s=(input('Enter PL,PW,SL & SW  ')) #enter 4 features of a sample input
temp = list(map(float, test_s.split())) #convert four inputs to list of four floats
test_s=np.array(temp)  #convert four inputs to an numpy array
setosa_o=setosa_classify(test_s,setosa_model_weights)
versicolor_o=versicolor_classify(test_s,versicolor_model_weights)
virginica_o=virginica_classify(test_s,virginica_model_weights)



first_layer_out=np.array([versicolor_o,setosa_o,virginica_o])

final_layer_out=np.ones(3)


final_layer_out[0]=logic1_classify(first_layer_out,logic1_model_weights)
final_layer_out[1]=logic2_classify(first_layer_out,logic2_model_weights)
final_layer_out[2]=logic3_classify(first_layer_out,logic3_model_weights)

#print(f'final_out1={final_layer_out[0]}')
#print(f'final_out2={final_layer_out[1]}')
#print(f'final_out3={final_layer_out[2]}')
if(versicolor_o==1 and setosa_o==0 and virginica_o==0):
    
    print(f'versicolor=({final_layer_out})')
elif(setosa_o==1 and virginica_o==0 and versicolor_o==0):
    
    print(f'setosa=({final_layer_out})')
elif(virginica_o==1 and versicolor_o==0 and setosa_o==0):
    
    print(f'virginica=({final_layer_out})')
else:
    print('misclassification')
    print(f'final_out={final_layer_out}')
    
    
    
    
'''   

#this part of the code for accuracy evaluation , if you want to 
       #get the accuracy of the whole network ,number of correct and miss classification
       #of the 150 sample , remove the quotation marks
       #the desired classification matrix and the setimated one will
       #be printed
       
       
def accuracy_evaluation_total(result,desired):   
    N_correct=0
    N_errors=0
    accuracy=0   
    for y in range (len(result)):
        for h in range(3):
          if(result[y][h]==desired[y][h]):
             N_correct=N_correct+1
   
          elif(result[y][h]!=desired[y][h]):
           N_errors=N_errors+1
    N_correct=N_correct/3
    N_errors=N_errors/3
    accuracy=(N_correct)/(len(desired)) *100
    
    return N_correct,N_errors,accuracy    
test_array=np.array([[0,1,0 ],[1,0,0],[0,0,1]])
network_estimation=np.array([[0,0,0 ],[0,0,0],[0,0,0]])

total_network_desired_out=np.repeat(test_array, repeats=50, axis=0)
total_network_test_result=np.repeat(network_estimation, repeats=50, axis=0)
print(total_network_desired_out)
test=np.array([0,0])
iris_data_matrix 
for x in range (len(iris_data_matrix)):
   setosa_o=setosa_classify(test_s,setosa_model_weights)
   versicolor_o=versicolor_classify(test_s,versicolor_model_weights)
   virginica_o=virginica_classify(test_s,virginica_model_weights)
   #virginica_o=virginica_classify(iris_data_matrix[x],virginica_model_weights)
   
   first_layer_out=np.array([setosa_o,versicolor_o,virginica_o])
   k=logic1_classify(first_layer_out,logic1_model_weights)
   l=logic2_classify(first_layer_out,logic2_model_weights)
   m=logic3_classify(first_layer_out,logic3_model_weights)
   total_network_test_result[x]=[k,l,m]
print(total_network_test_result)

correct,error,accuracy=accuracy_evaluation_total(total_network_test_result,total_network_desired_out)

print(f'err={error}')
print(f'crct={correct}')
print(f'acc={accuracy}')
'''