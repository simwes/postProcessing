import re
import numpy as np


def diff(a,b):

	for i in range(0, len(b)):
		for j in range(0, len(a)):
			if a[j] == b[i] :
				a[j] = None
	res = [] 
	for val in a: 
	    if val != None : 
	        res.append(val)
	return res

# Open a file
mylines = []
pointToRm = []
probe = []                          
with open ('log.simpleFoam', 'rt') as myfile: 
    for myline in myfile:                
        mylines.append(myline)


count = 1000000

for i in range(0, count):

	try:

		line = 68 + i*4
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

print('Number of points out of domain:', len(pointToRm))

for i in range(0, len(pointToRm)):
	print(pointToRm[i])

print(" ")
	
probe=[None]*8

probe[3]=pointToRm[0]
probe[4]=pointToRm[1]
probe[2]=pointToRm[2]
probe[0]=[0.5, 0.0, 1e-05]
probe[5]=[1.5, 0.0, 1e-05]
probe[1]=[2.2, 0.0, 1e-05]
probe[6]=[2.2, 10.0, 1e-05]
probe[7]=pointToRm[3]




for i in range(0, len(probe)):
	print(i, probe[i])

print('')


res = diff(probe, pointToRm)

print('')
for i in range(0, len(res)):
	print(i, res[i])