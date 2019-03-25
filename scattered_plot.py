print("Hello Anaconda")
import numpy as np
from matplotlib import pyplot as plt
x=np.array([0,0])
print(x)
y=np.array ([[1,1],[0,0]])
print(y)
print(y[0,0])
print(x[0])
a = np.array([1.5, 2, 3], dtype = float) 
b=np.array([5, 4, 6], dtype = float) 
c= np.array([10, 12, 13], dtype = float) 
d=np.array([15, 14, 16], dtype = float) 
print(a[0])




#plt.plot(a[0], b[0]) 
#plt.show()
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#for data, color in zip(data, colors):
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(a,b, s=10, c='b', marker="s", label='first')
ax1.scatter(c,d,s=10, c='r', marker="o", label='second')
plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()
