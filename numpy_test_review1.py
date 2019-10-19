#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: %(username)s
Email: jijunyu140@gmail.com
Github: https://github.com/yujijun
Description: 
    This scripts is for
Input:
Output:
    
"""
import numpy as np

#a example for numpy array create a simple array and find all attribute 
a = np.array([1,2,4])
type(a)
a.shape
a.ndim
a.size
a.itemsize
a.dtype

#array create 
a = np.array([1,2,3])
b = np.array((1,2,3))
c = np.array(((1,2),(3,4)))

d = np.zeros((2,3))
e = np.ones((2,3))
e = np.eye((3))
f = np.arange(12).reshape(3,4)
f_1 = np.arange(0,12,3)
g = np.linspace(0,2,5) #better, you can konw the number of sequence 
h = np.random.random((2,3))


#basic operation 
a = np.array([1,2,3,4])
b = np.arange(0,4,1)

a + b
a -b 
a**2
a/b

a = np.arange(4).reshape(2,2)
b = np.arange(2,6).reshape(2,2)
a * b
a @ b
a.dot(b)

np.exp(a)
np.sqrt(b)
#compare data # when you are use the function, you need to include bracket
a = np.random.random((2,3))
a
a.min()
a.max()
a.sum()

a.min(axis=0)
a.min(axis=1)
a.cumsum(axis=1)

#reshape
a = np.random.random((2,6))
a.ravel()
a.reshape(3,4)
a.resize(3,4)


#copy (always the same(data and shape))
a = np.arange(12)
b = a
b[3] = 100
a 

#view and shallow copy(just data is the same,shape could be different)
a = np.arange(12).reshape(3,4)
b = a.view()
b.shape = 2,6
a.shape
b[0,0] = 200
a 

#deep copy (a and b is completely different)
a
b = a.copy()
b.shape = 2,6
b[:,2] = 10
a

#stacking together different arrays
a = np.floor(10*np.random.random((2,2)))
b = np.arange(4).reshape(2,2)
np.hstack((a,b))
np.vstack((a,b))
np.r_[1:4,3,4]
np.c_[1:4,2:5]

#fancy index
#(1)array can be indexed by arrays of integers and arrays of booleans

a = np.arange(10)**3
a[0]
a[1:3]
a[::-1]
a[:6:2]
a[[0,0,1]] = [1,2,3] #last one will be replaced
a = np.arange(12)**2
i = np.array([1,2,2,8,5]) 
a[i] # order replication

j = np.array([[3,4],[9,7]])
a[j] #the same shape as j

#For example:
palette = np.array([[0,0,0],
                    [255,0,0],
                    [0,255,0],
                    [0,0,255],
                    [255,255,255]])
image = np.array([[0,1,2,0],
                  [0,3,4,0]])
palette[image]

#(2) two array
a = np.arange(12).reshape(3,4)
i = np.array([[0,1],
              [1,2]])
j = np.array([[2,1],
              [3,3]])
a[i,j]
l = [i,j]
a[l]
a[i,2]
a[:,j]

#(3)boolean
a = np.arange(12).reshape(3,4)
b1 = np.array([False,True,False])
a[b1,:]
a[b1]

#(4)how about 0/1
b1_1 = np.array([0,1,0])
b1_2 = np.array([0,1,1])
a[b1_1,:]
a[b1_2,:]

#(5)
# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
import matplotlib.pyplot as plt 
mu, sigma = 2, 0.5
v = np.random.normal(mu, sigma, 10000)
plt.hist(v,bins=50,density=1)
plt.show()

#compute the histogram with numpy and them plot it 
(n,bins) = np.histogram(v,bins=50,density=True)
plt.plot(.5*(bins[1:]+bins[:-1]),n)
plt.show()

