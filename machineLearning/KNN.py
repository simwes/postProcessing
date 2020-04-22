#======================================================================
# This script uses NearestCentroid to categorize new values in 2D grid
#
# X: grid points
# value: category values
# xPred: new point in the grid
# valuePred: predicted velue of the new grid point
#----------------------------------------------------------------------


from sklearn.neighbors import NearestCentroid
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt


X = np.array([ [0, 0.1], [1, 0], [2.4, 0], [0, 1], [1.6, 1], [2, 1] ])
value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#---- velue need to be encoded
le = preprocessing.LabelEncoder()
value_encoded = le.fit_transform(value)

model = NearestCentroid()
model.fit(X, value_encoded)

xPred = [1.6, 0.8]

valuePred = model.predict([ xPred ])

valuePredInv = le.inverse_transform( valuePred ) 

print( valuePredInv )


fig, ax = plt.subplots()

scatter = ax.scatter(X[:,0], X[:,1], c=value, s=50)
ax.scatter(xPred[0], xPred[1],  marker="x", label='New point', s=90)
plt.text(xPred[0]+0.08, xPred[1], r'%f' % valuePredInv, fontsize=10)

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(), loc="right", title="Value")
ax.add_artist(legend1)
ax.grid(True)

plt.legend()

plt.show()
