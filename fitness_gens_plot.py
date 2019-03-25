# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 01:20:50 2019

@author: Kasem
"""
from numpy import *
import numpy as np
from matplotlib import pyplot as plt
import pandas
df = pandas.read_excel('fitness vs gen.xls')


print (df.columns)


Generation = df['Generation'].values
Average_fitness = df['Average fitness:'].values
Best_fitness = df['Best fitness'].values

FORMAT = ['Generation', 'Average fitness:', 'Best fitness']
df_selected = df[FORMAT]
print('Generation')
print(Generation)
print('Average_fitness')
print(Average_fitness)
print('Best_fitness')
print(Best_fitness)

#fig0 = plt.figure()
#fig=fig0.add_subplot(111)
plt.xlabel('Generation')
plt.ylabel('Average_fitness')
plt.plot(Generation,Average_fitness)
plt.show()
plt.xlabel('Generation')
plt.ylabel('Best_fitness')
plt.plot(Generation,Best_fitness)
plt.show()



