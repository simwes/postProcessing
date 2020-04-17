from math import sqrt
import numpy as np
from scipy.stats import skew 
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import datetime
import math
import seaborn as sns
from statistics import mean

np.random.seed(7) 

# ----------- Interpolation -----------
def interpFunction( dt, nInt, xp, fp ):
			
	xpInt = np.zeros( nInt*(xp.size) + xp.size - nInt )

	for i in range(0, xp.size):
		for j in range(0,nInt+1):

			if i < xp.size-1:

				jInt =  j  + i*(nInt +1)
				xpInt[jInt] = xp[i] + j*dt/(nInt+1)

			else:

				j=0

				jInt =  j  + i*(nInt +1)
				xpInt[jInt] = xp[-1]

	return xpInt, np.interp(xpInt, xp, fp)


fishNumber= 'fish1.csv' 
dataOrg = pd.read_csv( fishNumber, sep = ",", skiprows = 1, header = None )
dataOrg = np.asarray( dataOrg )

Z=[]

dateTime = dataOrg[:, 0]

X = dataOrg[:, 1]
Y = dataOrg[:, 2]
for i in range(0, X.size):
	Z.append(15.5)

U = dataOrg[:, 4]
V = dataOrg[:, 5]	

uFlowXtot = dataOrg[:, 10]
uFlowYtot = dataOrg[:, 11]

# ------ Drag Force -----
FDx = dataOrg[:, 13]
FDy = dataOrg[:, 14]

# ----- Locomotion Force -----
FLx = dataOrg[:, 15] # < -------------- These need to be modeled
FLy = dataOrg[:, 16] #

nInt = 100
dt = 2
print( 'dt interp:', dt/(nInt+1) )


# # ------------ X component ------------
# length interplolated array:
lenOrg = X.size

uX   = np.zeros( lenOrg )
flX  = np.zeros( lenOrg )
fdX  = np.zeros( lenOrg )
xp   = np.zeros( lenOrg )

for i in range(0, lenOrg):
	uX[i] = uFlowXtot[i]	
	fdX[i] = FDx[i]
	flX[i] = FLx[i]

for i in range(0, lenOrg ):
	xp[i] = i*2

print( 'X Interpolation ...' )

xpInt, uFlowXtotIntp = interpFunction( dt, nInt, xp, uX )
xpInt, FLxIntp = interpFunction( dt, nInt, xp, flX )

# # -------------- Y component ------------

uY  = np.zeros( lenOrg )
flY = np.zeros( lenOrg )
fdY = np.zeros( lenOrg )
yp  = np.zeros( lenOrg )

for i in range( 0, lenOrg ):
	uY[i] = uFlowYtot[i]	
	fdY[i] = FDy[i]
	flY[i] = FLy[i] 


for i in range( 0, lenOrg ):
		yp[i] = i*2

print('Y Interpolation ...')

ypInt, uFlowYtotIntp = interpFunction( dt, nInt, yp, uY )
ypInt, FLyIntp = interpFunction( dt, nInt, yp, flY )

#----------------- Velocity -----------------------

print('X size', X.size, 'FDx size', FDx.size)

# length interplolated array:
lenInterp = xpInt.size

fdx   = np.zeros( lenInterp-1 )
flx   = np.zeros( lenInterp-1 )
ux    = np.zeros( lenInterp )
uXtot = np.zeros( lenInterp )

fdy   = np.zeros( lenInterp-1 )
fly   = np.zeros( lenInterp-1 )
uy    = np.zeros( lenInterp )
uYtot = np.zeros( lenInterp )

x = np.zeros( lenInterp )
y = np.zeros( lenInterp )


# ----------- Parameters for Drag coeficient ------------

dtInt =dt/(nInt+1)
mFish = 0.2 #kg 
rhoF = 1000
Df = 0.05
Cd = 0.044
coeff = ( Cd*math.pi*rhoF*Df**2 )/8.0

# -------- Initial Condition --------

ux[0] = U[0] 
uy[0] = V[0] 

x[0] = X[0] 
y[0] = Y[0] 


endTime = lenInterp - 1

print('Calculating velocity ...')


for t in range(0, endTime):
	
	fdx[t]  = -coeff*ux[t]*( ux[t]**2 + uy[t]**2 )**0.5
	flx[t]  =  FLxIntp[t]
	ux[t+1] =  ux[t] + ( fdx[t] + flx[t] )*dtInt/mFish 
	

	fdy[t]  = -coeff*uy[t]*( ux[t]**2 +  uy[t]**2 )**0.5 
	fly[t]  =  FLyIntp[t]
	uy[t+1] =  uy[t] + ( fdy[t] + fly[t] )*dtInt/mFish 


	#print( ux[t], uy[t], fdx[t], fdy[t], flx[t], fly[t] )


# --------  velocity with respect to fix reference -------
for t in range(0, endTime):
	
	uXtot[t] = ux[t] + uFlowXtotIntp[t]
	uYtot[t] = uy[t] + uFlowYtotIntp[t]

# -------- trajectory with respect to fix reference -------
for t in range(0, endTime):
	
	x[t+1] = x[t] + dtInt*uXtot[t]
	y[t+1] = y[t] + dtInt*uYtot[t]


pyplot.figure(1)
pyplot.title('Fish ')
pyplot.plot(x,y,'-o' ,color='red', label='prediction')
pyplot.plot(X,Y, color='blue', label='actual')
pyplot.legend()
pyplot.xlabel('X')
pyplot.ylabel('Y')

pyplot.show()
