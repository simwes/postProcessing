#==================================================================================
# This loads probes values of the three components of U 
# and store them in a 3D matrix, e.g. (time, U component, probe) .
# The input parameters are the name of the field 'U' and the number of probes.
#								Simone Olivetti
#===============================================================================

import matplotlib, pprint, time
import matplotlib.pyplot as plt
import numpy as np
import csv, re, json, math
import pandas as pd


#--- Probe file ----
fileU='U' 
#--- Grid points file ----
fileGrid='NewProbeGridPW.csv'

nDim = 3 # 3 dimensional
nProbes = 88536 # number of probes (any value)

print('Reading files ... ')

coord = pd.read_csv( fileGrid, sep = ",", skiprows = 1, header = None )
coord = np.asarray( coord )

X = coord[:, 0]
Y = coord[:, 1]
Z = coord[:, 2]


start_time = time.process_time()
df = pd.read_csv( fileU, sep = " ", skiprows = nProbes + 2, skipinitialspace = True )
df.columns = df.columns.str.strip()
print( 'Time to read:', time.process_time() - start_time, "seconds" )

nTsteps = len(df.iloc[:, 0])

print( 'nTsteps:', nTsteps )


print('Initializing initialize array ...')
start_time = time.process_time()

uTot = [ [[0.0 for k in range(nProbes)] for j in range(nDim)] for i in range(nTsteps) ]
uTot = np.asarray(uTot)
velMod = np.zeros( shape=( nProbes, nTsteps ) )
velMod = np.asarray(velMod)
timeArr = df.iloc[:, 0]
timeArr = np.asarray( timeArr )

print( 'Time to initialize array:', time.process_time() - start_time, "seconds" )


print('Composing arrays ...')
start_time = time.process_time()

for n in range(0, nProbes):
	
	u = df.iloc[ :, 1 + 3*n:4 + 3*n ]	
	u = np.asarray( u )

	for i in range(0, nTsteps):
		uTot[i,0,n] = float( u[i,0].replace('(', '') )
		uTot[i,1,n] = float( u[i,1] )
		uTot[i,2,n] = float( u[i,2].replace(')', '') )

		velMod[n,i] = np.sqrt( uTot[i,0,n]**2 + uTot[i,1,n]**2 + uTot[i,2,n]**2 )

print( 'Time to compose array:', time.process_time() - start_time, "seconds" )


print('Writting to file ...')
start_time = time.process_time()

for i in range(0, nTsteps):
	with open('uContour%d.csv' % i, 'w') as f:
			   writer = csv.writer( f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
			   writer.writerow(     [  'X', 'Y', 'Z', 'Velocity' ] )
			   writer.writerows( zip(   X,   Y,   Z,   velMod[:,i] ))

print( 'Time to write on file:', time.process_time() - start_time, "seconds" )


	for i in range(0,nTsteps):
		uTot[i,0,n] = float( u[i,0].replace('(', '') )
		uTot[i,1,n] = float( u[i,1] )
		uTot[i,2,n] = float( u[i,2].replace(')', '') )

for i in range(0,nTsteps):
	print( time[i], uTot[i,0,2], uTot[i,1,2], uTot[i,2,2] )
