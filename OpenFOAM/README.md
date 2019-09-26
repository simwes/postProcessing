###### readProbes.py:

This loads probes values of the three components of U  and store then in a 3D matrix, e.g. (time, U component, probe). 
The input parameters are the name of the field 'U' and the number of probes.

###### writeProbes.py:
This writes the list of probes in the system/controlDict file. It skips certain probes if required. It aslo write probes' coordinates in each probe file.
