import re, time, sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
import csv

def diff(a, b, epsilonMod, epsilonDir):

	# Here comparison meant as comparision between modules and angles

	pers = []

	for i in range(0,100,10):
		pers.append(i)

	aArrMod =[]
	aDir =[]
	bArrMod =[]
	bDir=[]

	aArr = np.asarray(a)
	bArr = np.asarray(b)
	
	for j in range( 0, len(a) ):
		aArrMod.append( np.sqrt( aArr[j,0]**2 + aArr[j,1]**2 ) )
		aDir.append(aArr[j,1]/aArr[j,0])

	for i in range( 0, len(b) ):
		bArrMod.append( np.sqrt( bArr[i,0]**2 + bArr[i,1]**2 ) )
		bDir.append(bArr[i,1]/bArr[i,0])
		

	for i in range( 0, len(b) ):
		for j in range(0, len(a)):
			#if a[j] == b[i] :
			if ( aArrMod[j] <= bArrMod[i] + epsilonMod and aArrMod[j] >= bArrMod[i] - epsilonMod and \
				 aDir[j] <= bDir[i] + epsilonDir and aDir[j] >= bDir[i] - epsilonDir ):	

				a[j] = None

		print( 'Progress %:', (i/len(b))*100 )
			

	res = [] 
	for val in a: 
	    if val != None : 
	        res.append(val)

	return res



def readFiles(logFileName, probeGridFileName, startToRead, skipLines, nSearch):
	# Open a file
	mylines = []
	pointToRm = []
	probe = []                          
	with open (logFileName, 'rt') as myfile: 
	    for myline in myfile:                
	        mylines.append(myline)


	count = nSearch

	for i in range(0, count):

		try:
			line = startToRead + i*skipLines
			s = mylines[line]
			result = re.search( 'Did not find location (.*) in any cell. Skipping location.', s )
			results = result.group(1)
			point=results.split(' ')

			point[0] = float( re.sub('[(]', ' ', point[0]) )
			point[1] = float( point[1] )
			point[2] = float( re.sub('[)]', ' ', point[2]) )		
			pointToRm.append( point )

		except:
		 	break

	pointToRmArr = np.asarray(pointToRm)

	print(" ")
	print('Reading original grid ...')

	data = pd.read_csv( probeGridFileName , sep = ",", skiprows = 1, header = None )
	data = np.asarray( data )


	X = data[:, 0]
	Y = data[:, 1]
	Z = data[:, 2]
	
	probe=[None]*X.size

	for i in range(0, len(probe)):
		probe[i] = [ X[i], Y[i], Z[i] ] 

	return probe, pointToRm, pointToRmArr

#===============================================================================================

#----------------------- Inputs ----------------------------
epsilonMod = 0.1 # tollerance for the module
epsilonDir = 0.1 # tollerance for the angle
startToRead = 84 # first line of the first outside domain point in log file (e.g. log.interFoam)
skipLines = 4 # lines to skip in log file (normaly 4)
nSearch = 5000000 # maximum number of outside domain points (wild guess)

logFileName = 'log.interFoam' # openfoam log file
probeGridFileName='probeGridPW.csv' # initial probe grid file. It cointains all the points

#----------------------- Outputs-------------------------
newGridFile = 'NewProbeGridPW.csv'

#--------------------------------------------------------


probe, pointToRm, pointToRmArr = readFiles(logFileName, probeGridFileName, startToRead, skipLines, nSearch)

res = diff( probe, pointToRm, epsilonMod, epsilonDir )

print('')
print('Number fo points original grid:', len(probe))
print('Number of points out of domain:', len(pointToRm))
print('Points final grid:', len(probe)- len(pointToRm) )

resArr = np.asarray(res)


print('')
print('writing new grid ...')

with open(newGridFile, 'w') as f:
	writer = csv.writer( f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
	writer.writerow( [ 'X', 'Y', 'Z' ] )
	writer.writerows( zip(  resArr[:,0], resArr[:,1], resArr[:,2] ) )

with open('pointsToRM.csv', 'w') as f:
	writer = csv.writer( f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
	writer.writerow( [ 'X', 'Y', 'Z' ] )
	writer.writerows( zip(  pointToRmArr[:,0], pointToRmArr[:,1], pointToRmArr[:,2] ) )

print('')
print('DONE!, new grid written in', newGridFile)

