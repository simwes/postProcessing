** readPoints.py execute the following operation **:  
1. Read coordinates from generic probes that may or may not be contained in the domain of computation. This is an arbitrary input file.
2. Read coordinates of probes outside the domain of computation from the openFOAM log file, e.g. log.interFoam. 
3. Remove the probes outside the domain of computation from the input file. See step 1.
4. Output a file containg anly the probes inside the domain of computation.
