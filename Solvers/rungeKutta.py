from math import sqrt
import numpy as np
from scipy.stats import skew 
from numpy import concatenate
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import datetime
import math
import seaborn as sns
from statistics import mean


def f( v, g, m, k):
    kmv = np.linalg.norm(v)*(k/m)
    return (g - kmv*v)


def RK4( xn, vn, h, g, m, k ):

    k1 = f( vn, g, m, k )
    k2 = f( vn + k1*h/2, g, m, k )
    k3 = f( vn + k2*h/2, g, m, k )
    k4 = f( vn + k3*h, g, m, k )
    vn1 = vn + (k1 + 2*k2 + 2*k3 + k4)*(h/6)

    k1x = vn
    k2x = vn + k1x*h/2
    k3x = vn + k2x*h/2
    k4x = vn + k3x*h
    xn1 = xn + (k1x + 2*k2x + 2*k3x + k4x)*(h/6)

    return vn1, xn1


theta = math.radians(10)
v0 = np.array([math.cos(theta), math.sin(theta)])
x0 = np.array([0, 0])
t0 = 0

print('theta:', theta*180/3.14)
print('v0:', v0)


x = []
v = []
t = []

xAnList =[]

x.append(x0)
v.append(v0)
t.append(t0)

m = 1
k = 0#0.01
g = np.array([0, -9.81])
h = 0.0001

tn = t0
vn = v0
xn = x0
maxIter = 2000

#while ( t < maxt ):
for i in range( 0, maxIter ):
	
	# ----------- Numerical --------
	vn, xn = RK4( xn, vn, h, g, m, k)	
	t.append(tn)
	x.append(xn)
	v.append(vn)

	# ----------- Analytical --------

	tn =  i*h

	xAn = 0.5*g *tn**2 + v0*tn + x0
	xAnList.append(xAn) 


print ('Final Time:', t[-1])

v   = np.asarray(v)
x   = np.asarray(x)
xAn = np.asarray(xAn)

xAnArr = np.asarray(xAnList)

print ('x:', x.shape)
print ('v:', v.shape)


plt.plot(x[:,0], x[:,1], label='KNN')
plt.plot(xAnArr[:,0], xAnArr[:,1], label='Actual')
plt.axis('equal')
plt.legend()
#plt.set(xlim=(0, 3), ylim=(0, 3))
plt.show()
