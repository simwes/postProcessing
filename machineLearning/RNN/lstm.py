import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import csv
# From here: https://www.tensorflow.org/tutorials/structured_data/time_series 

#====================================
# ----------- Input parameters ------

MAX_EPOCHS = 1 # Number of Epochs
windowSize = 4  # Window Size
valLenght  = 0.01  # Percentage of data for validation
OUT_STEPS  = 1
#------------------------------------
#====================================

print('')
print('tensorflow version', tf.version.VERSION)
print('')


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)


df = pd.read_csv(csv_path)
# slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')


print( df.head() )
print('')
print( df.describe().transpose() )

wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are reflected in the DataFrame
print('')
print(df['wv (m/s)'].min())

wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)')*np.pi / 180



# -------------------- Split the data ------------------
column_indices = { name: i for i, name in enumerate( df.columns ) }

row_indices = []
for i in range( 0, df.shape[0] ):
    row_indices.append(i)


n = len(df)
train_df = df[0:int(n*(1.-valLenght))]
val_df   = df[int(n*(1.-valLenght)):]
row_indices_df = row_indices[int(n*(1.-valLenght)):]

test_df  = df[int(n*0.9):]

num_features = df.shape[1]


print('Length', df.shape[0], df.shape[1] )


# --------- Normalize the data  -------
train_mean = train_df.mean()
train_std  = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df   =   (val_df - train_mean) / train_std
test_df  =  (test_df - train_mean) / train_std

print('train_mean', train_mean[1])
print('train_std', train_std[1])

data = np.asarray( df )
dataTest = np.asarray( test_df )
dataVal = np.asarray( val_df )

# with open('dataOutput.csv' ,  'w') as f:
#    writer = csv.writer( f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
#    writer.writerow( [     'T (degC)' ] )
#    writer.writerows( zip(   data[:,1] ) )

with open('dataOutputTest.csv' ,  'w') as f:
   writer = csv.writer( f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
   writer.writerow( [    'index', 'T (degC)' ] )
   writer.writerows( zip(  row_indices_df, dataVal[:,1]*train_std[1] + train_mean[1]  ) )


# ----- Data windowing -----
class WindowGenerator():
	def __init__(self, input_width, label_width, shift,
	           train_df=train_df, val_df=val_df, test_df=test_df, 
	           label_columns=None):
	    # Store the raw data.
	    self.train_df = train_df
	    self.val_df = val_df
	    self.test_df = test_df
	   

	    # Work out the label column indices.
	    self.label_columns = label_columns
	    if label_columns is not None:
	      self.label_columns_indices = {name: i for i, name in
	                                    enumerate(label_columns)}
	    self.column_indices = {name: i for i, name in
	                           enumerate(train_df.columns)}

	    # Work out the window parameters.
	    self.input_width = input_width
	    self.label_width = label_width
	    self.shift = shift

	    self.total_window_size = input_width + shift

	    self.input_slice = slice(0, input_width)
	    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

	    self.label_start = self.total_window_size - self.label_width
	    self.labels_slice = slice(self.label_start, None)
	    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]




	def __repr__(self):
	    return '\n'.join([
	        f'Total window size: {self.total_window_size}',
	        f'Input indices: {self.input_indices}',
	        f'Label indices: {self.label_indices}',
	        #f'Time indices: {self.time_indices}',
	        f'Label column name(s): {self.label_columns}'])

	def split_window(self, features):
		  inputs = features[:, self.input_slice, :]
		  labels = features[:, self.labels_slice, :]
		  if self.label_columns is not None:
		    labels = tf.stack(
		        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
		        axis=-1)

		  # Slicing doesn't preserve static shape information, so set the shapes
		  # manually. This way the `tf.data.Datasets` are easier to inspect.
		  inputs.set_shape([None, self.input_width, None])
		  labels.set_shape([None, self.label_width, None])

		  return inputs, labels



	def make_dataset(self, data):
	  data = np.array(data, dtype=np.float32)
	  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
	      data    = data,
	      targets = None,
	      sequence_length = self.total_window_size,
	      sequence_stride = 1,
	      shuffle = False,
	      batch_size = 2,)

	  ds = ds.map( self.split_window )

	  return ds	

	 



def compile_and_fit( model, window, patience = 2 ):
  early_stopping = tf.keras.callbacks.EarlyStopping( monitor  = 'val_loss',
                                                     patience = patience,
                                                     mode = 'min')

  model.compile( loss=tf.losses.MeanSquaredError(),
                 optimizer=tf.optimizers.Adam(),
                 metrics=[tf.metrics.MeanAbsoluteError()] )

  history = model.fit( window.train, epochs=MAX_EPOCHS,
                       validation_data=window.val,
                       callbacks=[early_stopping] )
  return history

	  
w2 = WindowGenerator( input_width=windowSize, label_width=windowSize, shift=OUT_STEPS, label_columns=['T (degC)'] )
print(w2)



# Add new function to WindowGenerator Class
@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result




WindowGenerator.train     = train
WindowGenerator.val       = val
WindowGenerator.test      = test
WindowGenerator.example   = example



lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense( units=1 )
])


history = compile_and_fit( lstm_model, w2 )

IPython.display.clear_output()

val_performance = {}
performance = {}


val_performance['LSTM'] = lstm_model.predict(w2.val)
outPred = lstm_model.predict(w2.val)



outPred=np.asarray( outPred )
print('outPred:', outPred.shape)

lastCell = windowSize - 1


predList, ref = [], []
for i in range(0, outPred[:,lastCell].size):
	predList.append( outPred[i,lastCell,0]*train_std[1] + train_mean[1] )

refArr=np.asarray( dataVal[len(row_indices_df)-len(predList):,1] )
print('refArr:', refArr.shape,train_std , train_mean)
	

with open('outPred.csv' ,  'w') as f:
   writer = csv.writer( f, delimiter=',', quoting=csv.QUOTE_MINIMAL )
   writer.writerow( [  'index', 'prediction' ] )
   writer.writerows( zip( row_indices_df[len(row_indices_df)-len(predList):-windowSize], predList ) )


