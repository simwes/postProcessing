import matplotlib, pprint
import matplotlib.pyplot as plt
import numpy as np
import csv, re, json, math
import pandas as pd


file='U'

nDim=4
nProbes=3


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
