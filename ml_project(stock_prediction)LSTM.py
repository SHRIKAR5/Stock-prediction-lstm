key=' ' # get key from tingo by crating account there


### Keras and Tensorflow >2.0
### Data Collection
import pandas_datareader as pdr
df = pdr.get_data_tiingo('AAPL', api_key=key)
df.to_csv(r'C:\...\AAPL.csv')  # save file to a location

import pandas as pd
df=pd.read_csv(r'C:\.....\AAPL.csv') # call the same file
df.head()
df.tail()
df2=df.reset_index()['close']
df3=df.reset_index()['close']

df2.shape
#df2=df['close']
#df2
import matplotlib.pyplot as plt
plt.plot(df2)
plt.plot(df3)

#f=df.loc[1140:1160,]
#------------------------------------------------------------------------------------------------

### LSTM are sensitive to the scale of the data. so we apply MinMax scaler
import numpy as np
df2
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
#we .reshape(-1,1) df2 since scaler.fit_transform requires 2d
df2=scaler.fit_transform(np.array(df2).reshape(-1,1))
print(df2)
len(df2)
##splitting dataset into train and test split

training_size=int(len(df2)*0.7)
test_size=len(df2)-training_size
train_data= df2[0:training_size]
train_data
len(train_data)
test_data=df2[training_size:]
test_data
len(test_data)


#len(train_data) = len(dataset)
#880
import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):         #880-150-1 = 729
        #                            ^reshape(1,-1)
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)
# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 150
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

print(X_train.shape), print(y_train.shape)
#X_train[0].shape
#X_train[715].shape
y_train[0]
print(X_test.shape), print(ytest.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(150,1)))
#                    from X_train        (no. of columns,1)
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))                                                 #can we use our type?
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()
model.summary()

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

import tensorflow as tf
tf.__version__
#'2.1.0'

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))
### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))

#------------------------------------------------------------------------------------------------

look_back=150
trainPredictPlot = numpy.empty_like(df2)
trainPredictPlot[:, :] = np.nan
trainPredictPlot.shape
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df2)
testPredictPlot[:, :] = numpy.nan/9
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df2)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df2))
#plt.plot(train_predict)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

#we store last 150 days data from 1257 in a x_input to predict next 100 days data<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
len(test_data)
#378
378-150          
#       
x_input=test_data[228:].reshape(1,-1)
test_data.shape
x_input.shape
#---------------------------------------------------------------------------------------------

temp_input1=x_input[0]
temp_input1.shape
temp_input1=x_input[0].tolist()
temp_input1.shape
#---------------------------------------------------------------------------------------------

temp_input=x_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=150
i=0
while(i<100):
    
    if(len(temp_input)>150):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        #print(yhat[0])  approxes 5 digits after 0. to get full 17pts after 0. we use .tolist()
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    
print(lst_output)

day_new=np.arange(1,151)
day_pred=np.arange(151,251)
251*2
import matplotlib.pyplot as plt

len(df2)
1258-150
plt.plot(day_new,scaler.inverse_transform(df2[1108:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))

df3=df2.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])

df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)
    

 [1 [2 3 4 5] 6 ]7 8 9 10



    
    
    
    
    
    
    
    
    
    
    

