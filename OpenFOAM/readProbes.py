#==================================================================================
# This loads probes values of the three components of U 
# and store then in a 3D matrix, e.g. (time, U component, probe) .
# The input parameters are the name of the field 'U' and the number of probes.
#								Simone Olivetti
#===============================================================================

import matplotlib, pprint
import matplotlib.pyplot as plt
import numpy as np
import csv, re, json, math
import pandas as pd


#--- Inputs ----
file='U' # file name

nDim=3 # 3 dimensional
nProbes=3 # number of probes (any value)


df = pd.read_csv(file, sep=" ", skiprows=nProbes+2, skipinitialspace=True)
df.columns = df.columns.str.strip()

nTsteps = len(df.iloc[:, 0])

uTot = [[[0.0 for k in range(nProbes)] for j in range(nDim)] for i in range(nTsteps)]
uTot = np.asarray(uTot)

time = df.iloc[:, 0]
time = np.asarray(time)

for n in range(0, nProbes):
	
	u = df.iloc[:, 1 + 3*n:4 + 3*n ]
	
	u = np.asarray(u)

	for i in range(0,nTsteps):
		uTot[i,0,n] = float( u[i,0].replace('(', '') )
		uTot[i,1,n] = float( u[i,1] )
		uTot[i,2,n] = float( u[i,2].replace(')', '') )

for i in range(0,nTsteps):
	print( time[i], uTot[i,0,2], uTot[i,1,2], uTot[i,2,2] )
