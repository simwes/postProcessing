import matplotlib, pprint
import matplotlib.pyplot as plt
import numpy as np
import csv, re, json, math
import pandas as pd

# Index to skip
skipIndex=[ 10, 23, 29, 32, 46, 63, 73, 17, 27, 34, 88, 135 ]


#=============== Write list of probes in System file ===============

# read  input file into a variable
with open('controlDict', 'r') as f:
    in_file = f.readlines()  # in_file is now a list of lines

# building our output
out_file = []
for line in in_file:
    out_file.append(line)  # copy each line, one by one

    if 'functions' in line:  # add a new entry, after a match
        out_file.append('{ \n')
        
        for i in range(0, 197):
        	out_file.append(' #includeFunc probe%d \n' % i )

        	if i in skipIndex:
        		out_file.remove(' #includeFunc probe%d \n' % i)
        		print('Skip: ',i)
        	
        out_file.append('} \n')

with open('new_controlDict', 'w') as f:
    f.writelines(out_file)

#=============== Write coordinates for each probe ===============


in_file = [ '#includeEtc "caseDicts/postProcessing/probes/probes.cfg"' , 'setFormat   csv;', 'fields (U);', 'probeLocations (', ')' ]

for nFishCount in range(0,10):

	data = pd.read_csv( 'TagsTrans/fish%d.csv' % nFishCount, sep = ",", skiprows = 1, header = None )
	data = np.asarray( data )

	leftBr=[]
	rightBr=[]


	time = data[:, 0]
	dateTime = data[:, 1]
	X = data[:, 2]
	Y = data[:, 3]
	Z = data[:, 4]

	for i in range(0, X.size):
		leftBr.append('  (')
		rightBr.append(')')


	with open('probe%d' % nFishCount, 'w') as f:
	    writer = csv.writer(f, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

	    writer.writerow([in_file[0], '\n', in_file[1], '\n', in_file[2],  '\n', in_file[3] ])
	    writer.writerows( zip( leftBr, X, Y, Z, rightBr  ) )
	    writer.writerow( [in_file[4]] )
	    






