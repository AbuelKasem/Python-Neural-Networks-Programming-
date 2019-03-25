# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:48:06 2019

@author: Kasem
"""
from numpy import *
import numpy as np
from matplotlib import pyplot as plt
col1 = []
col2 = []
col3 = []
f = open("iris.txt", "r")
#data_in=loadtxt('iris.dat')
#print(f.read())


sepal_length,sepal_width,petal_length,petal_width=loadtxt('iris.dat',unpack=True)
   
print(sepal_length)
print(sepal_width)
print(petal_width)
print(petal_length)
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(sepal_length,sepal_width, s=10, c='b', marker="s", label='s_l vs s_w')
ax1.scatter(petal_length,petal_width,s=10, c='r', marker="o", label='p_l vs p_w')
plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()
#print(data[:,1])
