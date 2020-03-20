import numpy as np
from keras.layers import Dense
from keras.models import Sequential

# Read the data
predictors = np.loadtxt('predictors_data.csv', delimiter=', ')
n_cols = predictors.shape[1]

# Build the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols, )))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
