# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:48:06 2019

@author: Kasem
"""
from numpy import *
import numpy as np
from matplotlib import pyplot as plt


#data_in=loadtxt('iris.dat')
#print(f.read())
#setosa_sl=np.array
   #loading the four inputs from data set into four separate arrays 
sepal_length,sepal_width,petal_length,petal_width=loadtxt('iris.dat',unpack=True)
   #slicing each array for each iris
setosa_s=slice(0,50,1)
versicolor_s=slice(50,100,1)
virginica_s=slice(100,150,1)

 
           #assigning the values of setosa to 4 arrays for each input
setosa_sl=sepal_length[setosa_s] 
setosa_sw=sepal_width[setosa_s] 
setosa_pl=petal_length[setosa_s] 
setosa_pw=petal_width[setosa_s]
             #assigning the values of versicolor to 4 arrays for each input
versicolor_sl=sepal_length[versicolor_s] 
versicolor_sw=sepal_width[versicolor_s] 
versicolor_pl=petal_length[versicolor_s] 
versicolor_pw=petal_width[versicolor_s]
             #assigning the values of virginica to 4 arrays for each input
virginica_sl=sepal_length[virginica_s] 
virginica_sw=sepal_width[virginica_s] 
virginica_pl=petal_length[virginica_s] 
virginica_pw=petal_width[virginica_s]

#print(sepal_length[0:50])
                          # to display  input data uncomment the 3 quotations marks
'''
print('setosa_sl')
print(setosa_sl)
print('setosa_sw')
print(setosa_sw)
print('setosa_pl')
print(setosa_pl)
print('setosa_pw')
print(setosa_pw)
    #display versicolor input data
print('versicolor_sl')
print(versicolor_sl)
print('versicolor_sw')
print(versicolor_sw)
print('versicolorpl')
print(versicolor_pl)
print('versicolor_pw')
print(versicolor_pw)
   #display virginica input data
print('virginica_sl')
print(virginica_sl)
print('virginica_sw')
print(virginica_sw)
print('virginicar_pl')
print(virginica_pl)
print('virginica_pw')
print(virginica_pw)
#print('sepal_length')
#print(sepal_length)
#print(sepal_width)
#print(petal_width)
#print(petal_length)
                             '''
                       #figure 1
fig0 = plt.figure()
fig=fig0.add_subplot(111)
#petal = fig.add_subplot(111)
#sepal = fig.add_subplot(111)
fig.scatter(setosa_sw,setosa_sl, s=10, c='b', marker="s", label='setosa')
fig.scatter(versicolor_sw,versicolor_sl,s=10, c='r', marker="o", label='versicolor')
fig.scatter(virginica_sw,virginica_sl,s=10, c='g', marker="o", label='virginica')
fig.scatter(setosa_pw,setosa_pl, s=10, c='b', marker="s")
fig.scatter(versicolor_pw,versicolor_pl,s=10, c='r', marker="o")
fig.scatter(virginica_pw,virginica_pl,s=10, c='g', marker="o")


plt.title('Iris data set P & S  widths vs P & S length')
plt.legend(loc=2)
plt.plot([0,4.5],[1.8,6.5])   #separation line plot
plt.rc('grid', linestyle="-", color='black')
plt.ylabel('P & S length')
plt.xlabel('P & S  widths')
plt.grid(True)
plt.show()

                 # plots

                  # plotting petal width  vs petal  length
fig1 = plt.figure()
widthl1 = fig1.add_subplot(111)
widthl1.scatter(setosa_pw,setosa_pl, s=10, c='b', marker="s", label='setosa')
widthl1.scatter(versicolor_pw,versicolor_pl,s=10, c='r', marker="o", label='versicolor')
widthl1.scatter(virginica_pw,virginica_pl,s=10, c='g', marker="o", label='virginica')
plt.title(' Petal Width  vs Petal  Length')
plt.legend(loc=2)
plt.plot([0,2],[3,0])
plt.rc('grid', linestyle="-", color='black')
plt.ylabel('Petal  Length')
plt.xlabel('Petal Width')
plt.grid(True)
plt.show()
                 

           
# plotting petal width vs sepal width
fig2 = plt.figure()
widthl = fig2.add_subplot(111)
widthl.scatter(setosa_pw,setosa_sw, s=10, c='b', marker="s", label='setosa')
widthl.scatter(versicolor_pw,versicolor_sw,s=10, c='r', marker="o", label='versicolor')
widthl.scatter(virginica_pw,virginica_sw,s=10, c='g', marker="o", label='virginica')
plt.title('Petal Width vs Sepal Width')
plt.legend(loc=2)
plt.plot([0,4.5],[1.8,6.5])  #separation line plot
plt.rc('grid', linestyle="-", color='black')
plt.ylabel('Sepal Width')
plt.xlabel('Petal Width')
plt.grid(True)
plt.show()
                        # plotting petal width  vs sepal  length
fig3 = plt.figure()
widthl1 = fig3.add_subplot(111)
widthl1.scatter(setosa_pw,setosa_sl, s=10, c='b', marker="s", label='setosa')
widthl1.scatter(versicolor_pw,versicolor_sl,s=10, c='r', marker="o", label='versicolor')
widthl1.scatter(virginica_pw,virginica_sl,s=10, c='g', marker="o", label='virginica')
plt.title('Petal Width  vs Sepal  Length ')
plt.legend(loc=2)
plt.plot([0,1.1],[2.3,6.9])  #separation line plot
plt.rc('grid', linestyle="-", color='black')
plt.ylabel('Sepal Length')
plt.xlabel('Petal Width')
plt.grid(True)
plt.show()

# plotting sepal width  vs petal  length
fig4 = plt.figure()
widthl = fig4.add_subplot(111)
widthl.scatter(setosa_sw,setosa_pl, s=10, c='b', marker="s", label='setosa')
widthl.scatter(versicolor_sw,versicolor_pl,s=10, c='r', marker="o", label='versicolor')
widthl.scatter(virginica_sw,virginica_pl,s=10, c='g', marker="o", label='virginica')
plt.title('Sepal Width  vs Petal  Length ')
plt.legend(loc=2)
plt.plot([0,5],[4.5,5.5])  #separation line plot
plt.plot([0,5],[2,3])  #separation line plot
plt.rc('grid', linestyle="-", color='black')
plt.ylabel('Petal  Length')
plt.xlabel('Sepal Width')
plt.grid(True)
plt.show()
                        # plotting sepal width  vs sepal  length
fig6 = plt.figure()
widthl1 = fig6.add_subplot(111)
widthl1.scatter(setosa_sw,setosa_sl, s=10, c='b', marker="s", label='setosa')
widthl1.scatter(versicolor_sw,versicolor_sl,s=10, c='r', marker="o", label='versicolor')
widthl1.scatter(virginica_sw,virginica_sl,s=10, c='g', marker="o", label='virginica')
plt.title('Sepal Width  vs Sepal  Length ')
plt.legend(loc=2)

plt.plot([0,4.5],[2,7])  #separation line plot
plt.rc('grid', linestyle="-", color='black')
plt.ylabel('Sepal  Length')
plt.xlabel('Sepal Width')
plt.grid(True)
plt.show()


# plotting  petal  length  vs sepal  length
fig7 = plt.figure()
widthl1 = fig7.add_subplot(111)
widthl1.scatter(setosa_pl,setosa_sl, s=10, c='b', marker="s", label='setosa')
widthl1.scatter(versicolor_pl,versicolor_sl,s=10, c='r', marker="o", label='versicolor')
widthl1.scatter(virginica_pl,virginica_sl,s=10, c='g', marker="o", label='virginica')
plt.title('Petal  Length  vs Sepal  Length ')
plt.legend(loc=2)
plt.plot([0,4],[1,8])   #separation line plot
plt.rc('grid', linestyle="-", color='black')
plt.ylabel('Sepal  Length')
plt.xlabel('Petal  Length ')
plt.grid(True)
plt.show()
