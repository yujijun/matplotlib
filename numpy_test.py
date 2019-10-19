#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:33:39 2019
This is the test of the numpy
@author: yujijun
"""
import numpy as np 
# a simple example of numpy 
a = np.arange(15).reshape(3,5)
a.shape
a.ndim
a.dtype.name
a.itemsize
a.size
type(a)

#array create 
a = np.array([1,2,3])
b =np.array((1,2,3))
np.zeros((3,4))
np.ones((3,4,5))
np.empty((2,3))
np.arange(10,20,5)
np.arange(1,3,0.3)
np.linspace(2,4,20)
np.random.random((2,3))
x =np.linspace(0,2*np.pi,100)
y = np.sin(x)

#basic operations
a = np.array([10,20,30,40])
b = np.arange(4)
c = a -b 
b**2
b*np.sin(a)
a < 35

A = np.arange(4).reshape(2,2)
B = np.arange(2,6).reshape(2,2)
A*B
A@B
A.dot(B)

a = np.random.random((2,3))
a.sum()
a.min()
a.max()

b = np.arange(12).reshape(3,4)
b.sum(axis=0)
b.min(axis=1)
b.cumsum(axis=1)

#universal function 
B = np.arange(3)
np.exp(B)
np.sqrt(B)



#reshape
a.ravel()
a.reshape(3,4)
a.resize(3,4)


#stacking together different arrays
a = np.floor(10*np.random.random((2,2)))
b = np.floor(10*np.random.random((2,2)))
c = np.floor(10*np.random.random((2,2)))
np.vstack((a,b,c))
np.hstack((a,b,c))
np.r_[1:4,3,4]
np.c_[1:4,3:6]


#hsplit
a = np.floor(10*np.random.random((2,12)))
np.hsplit(a,3)
np.hsplit(a,(3,5))


#copy and view 
# no copy at all (always the same)
a = np.arange(12)
b = a  #a and b are two names for the same ndarray object
b is a 
b.shape = 3,4
a.shape

#view and shallow copy (just data is the same)
#shape could change but two dataset still the same
c = a.view()
c is a 
c.base is a #True
c.shape = 2,6
a.shape
c[0,4] = 1234
a
#slicing an array returns a view of it :
s = a[:,1:3]
s[:] = 10 #Note the difference between s=10 and s[:]=10
a
 
#Deep copy (all isn't same)
d = a.copy() #a new array object with new data is created
d is a
d.base is a  #d doesn't share anything with a

#Sometimes copy should be called after slicing if the original array is not required anymore. For example, suppose a is a huge intermediate result and the final result b only contains a small fraction of a, a deep copy should be made when constructing b with slicing:
a = np.arange(int(1e8))
b = a[:100].copy()
del a  # the memory of ``a`` can be released.
#If b = a[:100] is used instead, a is referenced by b and will persist in memory even if del a is executed.

a = np.arange(12).reshape(3,4)
b1 = np.array([0,1,1])
b2 = np.array([False,True,True])
a[b1,:]
a[b2,:]


#fancy indexing and index tricks

#(1)
#arrays can be indexed by arrays of integers and arrays of booleans
#indexing, slicing  and interating
a = np.arange(10)**3
a[2]
a[2:5]
a[::-1]
a[:6:2] = -1000
a = np.arange(12)**2
i = np.array([1,1,2,8,5])
a[i]
j = np.array([[3,4],[9,7]]) # a bidimensional array of indices
a[j] # the same shape as j 

#for example 
palette = np.array( [ [0,0,0],                # black
                      [255,0,0],              # red
                      [0,255,0],              # green
                      [0,0,255],              # blue
                      [255,255,255] ] )   

image = np.array([[0,1,2,0],[0,3,4,0]])
palette[image]

#(2)
a = np.arange(12).reshape(3,4)
a
i = np.array([[0,1],
              [1,2]])
j = np.array([[2,1],
              [3,3]])
a[i,j] 
l = [i,j]
a[l]
a[i,2] #broad
a[:,j] #: same as 1,2,3

#(3)
a = np.arange(5)
a[[1,3,4]] = 0
a

a = np.arange(5)
a[[0,0,2]] = [1,2,3] #when the list of indices contains repetitions, the assignment is done several times, leaving 
#behind the last value
a


#(4)
#indexing with boolean arrays
a = np.arange(12).reshape(3,4)
b = a > 4 
a[b] #1d array with the selected elements
a[b] = 0 #all elements of "a" higher than 4 become 0
a

#(5) 
a = np.arange(12).reshape(3,4)
b1 = np.array([False, True,True])
b2 = np.array([True,False,True,False])
a[b1,:]
a[b1]
a[:,b2]
a[b1,b2]

import numpy as np
import matplotlib.pyplot as plt
# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
mu, sigma = 2, 0.5
v = np.random.normal(mu,sigma,10000)
# Plot a normalized histogram with 50 bins
plt.hist(v, bins=50, density=1)       # matplotlib version (plot)
plt.show()

# Compute the histogram with numpy and then plot it
(n, bins) = np.histogram(v, bins=50, density=True)  # NumPy version (no plot)
plt.plot(.5*(bins[1:]+bins[:-1]), n)
plt.show()
