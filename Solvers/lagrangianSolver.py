from math import sqrt
import numpy as np
from scipy.stats import skew 
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import datetime, re
import math
import seaborn as sns
from statistics import mean

np.random.seed(7) 

# ----------- Interpolation -----------
def interpFunction( dt, nInt, xp, fp ):
			
	xpInt = np.zeros( nInt*xp.size + xp.size - nInt )

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


# ------------ Parameters ---------------
iFish = 86

dt = 2
nIntF  = 100
dtIntF = dt/(nIntF + 1)

print( 'dt interpF:', dt/(nIntF + 1) )
#-------------------------------------


fishNumber= 'fish%d.csv' % iFish 
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

index=[]
index.append(0)
hour   =  np.zeros( dataOrg[:, 1].size )
minute =  np.zeros( dataOrg[:, 1].size )
second =  np.zeros( dataOrg[:, 1].size )
sec    =  np.zeros( dataOrg[:, 1].size )

for i in range( 0, len(dateTime) ): 

		dateTimeSplit = re.split( "\s", dateTime[i], 1 )
		time      = re.split( ":", dateTimeSplit[1] )
		hour[i]   = time[0]
		minute[i] = time[1]
		second[i] = time[2]

		sec[i] = hour[i]*3600 + minute[i]*60 + second[i]	

		if ( i > 0 and sec[i] != sec[i-1] + 2 ):
			index.append(i)

index.append( dataOrg[:, 1].size - 1 )


# # ------------ X component ------------
lenOrg = X.size

uX   = np.zeros( lenOrg )
flX  = np.zeros( lenOrg )
fdX  = np.zeros( lenOrg )
xp   = np.zeros( lenOrg )

for i in range(0, lenOrg ):
	xp[i] = i*dt

# -------------- Y component ------------

uY  = np.zeros( lenOrg )
flY = np.zeros( lenOrg )
fdY = np.zeros( lenOrg )
yp  = np.zeros( lenOrg )

for i in range( 0, lenOrg ):
		yp[i] = i*dt


#----------------- Velocity -----------------------
fdx   = np.zeros( lenOrg-1 )
flx   = np.zeros( lenOrg-1 )
ux    = np.zeros( lenOrg )
uXtot = np.zeros( lenOrg )
uYtot = np.zeros( lenOrg)

fdy   = np.zeros( lenOrg-1 )
fly   = np.zeros( lenOrg-1 )
uy    = np.zeros( lenOrg )


# ----------- Parameters for Drag coeficient ------------
mFish = 0.2 #kg 
rhoF = 1000
Df = 0.05
Cd = 0.044
coeff = ( Cd*math.pi*rhoF*Df**2 )/8.0

# -------- Initial Condition --------
ux[0] = U[0] 
uy[0] = V[0] 

endTime = lenOrg - 1

print('Calculating velocity ...')

#------------------- forward  --------------
for t in range(0, endTime):
	
	fdx[t]  = -coeff*ux[t]*( ux[t]**2 + uy[t]**2 )**0.5
	flx[t]  =   FLx[t]
	ux[t+1] =   ux[t] + ( fdx[t] + flx[t] )*dt/mFish 
	

	fdy[t]  = -coeff*uy[t]*( ux[t]**2 +  uy[t]**2 )**0.5 
	fly[t]  =   FLy[t]
	uy[t+1] =   uy[t] + ( fdy[t] + fly[t] )*dt/mFish 


# # --------  velocity with respect to fix reference -------
for t in range( 0, endTime ):
	
	uXtot[t] = ux[t] + uFlowXtot[t]
	uYtot[t] = uy[t] + uFlowYtot[t]

# -------- trajectory with respect to fix reference ------
xpIntF, uXtotInt = interpFunction( dt, nIntF, xp, uXtot )
ypIntF, uYtotInt = interpFunction( dt, nIntF, yp, uYtot )

lenInterpF = xpIntF.size
endTimeF = lenInterpF - 1

x = np.zeros( lenInterpF )
y = np.zeros( lenInterpF )

x0=[]
y0=[]

# -------------- Select chunks index -----------
for iChank in range( 0, len(index) ):
		
		if iChank == 0:
			iCi =0
		else:
			iCi =iChank-1

		for i in range(index[iCi], index[iChank]):
			if i == index[iCi] :
				
				x0.append( X[index[iCi]] )
				y0.append( Y[index[iCi]] )
				
print('Number of chunks:', len(index)-1)

#--------------------------------------------

x[0] = x0[0]
y[0] = y0[0]

for t in range(0, endTimeF):
	# ------- update initial condition for each chunk --------
	for i in range( 1, len(index) -1 ):
		if (t  == (nIntF +1)*index[i]  ):
			x[t] = x0[i]
			y[t] = y0[i]
	
	x[t+1] = x[t] + dtIntF*uXtotInt[t]
	y[t+1] = y[t] + dtIntF*uYtotInt[t]




pyplot.figure(1)
pyplot.title('Fish %d' % iFish)
pyplot.plot(x,y,'-' ,color='red', label='prediction')
pyplot.plot(X,Y, color='blue', label='actual')
pyplot.legend()
pyplot.xlabel('X')
pyplot.ylabel('Y')


pyplot.figure(2)
pyplot.subplot(211)
pyplot.title('U vs ux ')
pyplot.plot( xp, U,'-o' ,color='blue', label='actual')
pyplot.plot( xp, ux,'-' ,color='red', label='Integration')

pyplot.legend()

pyplot.subplot(212)
pyplot.title('V vs uy ')
pyplot.plot( yp, V, '-o' ,color='blue', label='actual')
pyplot.plot( yp, uy, '-' ,color='red', label='Integration')

pyplot.legend()



pyplot.show()
