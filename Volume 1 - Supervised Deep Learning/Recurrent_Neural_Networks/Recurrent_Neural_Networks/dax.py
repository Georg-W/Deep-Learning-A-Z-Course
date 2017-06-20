# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('Dax_max.csv')
data = data.iloc[:,1:2].values

counter = 0
for item in data:
    print("out", item)
    if data[counter][0] != 'null':
        print("in", item)
        data[counter][0]=float(data[counter][0])
    else:
        data[counter][0]=0
    counter += 1

#Feature Scaling Normalization
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
data = sc.fit_transform(data)

#Getting Inputs and Outputs
X_train = data[0:4000]
y_train = data[1:4001]


X_train = np.reshape(X_train, (4000, 1, 1))

#Init the RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor = Sequential()

#input layer with Long-Shortterm Memory
regressor.add(LSTM(units = 20, activation = 'sigmoid', input_shape = (None, 1)))

#output layer
regressor.add(Dense(units = 1))

#Compiling

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 200, batch_size = 32)

#prediction
test_set = pd.read_csv('Dax_max.csv')
real_stock_price = test_set.iloc[:,1:2].values

counter = 0
for item in real_stock_price:
    print("out", item)
    if real_stock_price[counter][0] != 'null':
        print("in", item)
        real_stock_price[counter][0]=float(real_stock_price[counter][0])
    else:
        real_stock_price[counter][0]=0
    counter += 1
    
real_stock_price = real_stock_price[4001:6000]
real_stock_price = real_stock_price[4002:6001]
    
#Google 2017 prediction
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (1999, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#Visualizing
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#Evaluate

import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

#around average of values dealt with (800), under 1% is great
score = rmse/800







