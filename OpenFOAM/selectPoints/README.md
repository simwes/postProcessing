**readPoints.py executes the following operations**:  
1. Read coordinates from generic probes that may or may not be contained in the domain of computation. This is an arbitrary input file.
2. Read coordinates of probes outside the domain of computation from the openFOAM log file, e.g. log.interFoam. 
3. Remove the probes outside the domain of computation from the input file. See step 1.
4. Output a file containg only the probes inside the domain of computation.

**Visual description:**
![test-page-001](https://user-images.githubusercontent.com/36754185/78853334-7cf96180-79d3-11ea-80e8-66fca3439470.jpg)
**Run example:**

*python3 readPoints.py*

Read generic probes from inputProbes.csv; read outside domain points from log.interFoam; output only inside domain probes. 
