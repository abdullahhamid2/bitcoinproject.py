#THIS IS A GROUP PROJECT FOR CS 334. CONTRIBUTORS INCLUDE: ENDER SHMIDT, ABDULLAH HAMID AND TULIO CANO
import argparse
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dropout
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    dataSet = pd.read_csv("bitcoinUSD.csv")
    dataSet.dropna(subset = ["Open"], inplace = True)
    dataSet["Timestamp"] = pd.to_datetime(dataSet["Timestamp"], unit = 's').dt.date
    dataSet = dataSet.groupby("Timestamp", as_index = False).mean() # comment this out for full data set.
    ySet = dataSet["Close"]
    ySet2 = dataSet["Open"] # Change prediction feature here
    ySet = np.array(ySet)
    ySet2 = np.array(ySet2)
    tickNum = 50
    trainSize = round(len(ySet) * 0.75) # Change training size here
    yTrain = ySet2[:trainSize]
    yTest = ySet[trainSize:]
    yActualTest = ySet[trainSize:]
    scaler = MinMaxScaler()
    yTrain = yTrain.reshape(-1, 1)
    yTrain = scaler.fit_transform(yTrain)
    yTrainOne = yTrain[0:len(yTrain)-1]
    yTrainTwo = yTrain[1:len(yTrain)]
    yTrainOne = np.reshape(yTrainOne, (yTrainOne.shape[0], yTrainOne.shape[1], 1))
    regressor = Sequential()
    function = "relu"
    regressor.add(SimpleRNN(units = 200, activation=function, return_sequences = True, input_shape = (1, 1)))
    #Adds dropouts and extra RNN layers.
    #regressor.add(Dropout(0.2))
    #regressor.add(SimpleRNN(units = 200, activation=function, return_sequences = True))
    #regressor.add(Dropout(0.2))
    #regressor.add(SimpleRNN(units = 200, activation=function, return_sequences = True))
    #regressor.add(Dropout(0.2))
    regressor.add(SimpleRNN(units = 200))
    #regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') # Change optimizer and loss here.
    regressor.fit(yTrainOne, yTrainTwo, epochs = 10, batch_size = 200) # Change epochs and batch size here.
    yTestInput = np.reshape(yTest, (yTest.shape[0], 1))
    yTestInput = scaler.transform(yTestInput)
    yTestPred = regressor.predict(yTestInput)
    yTestPred = scaler.inverse_transform(yTestPred)
    plt.figure(figsize=(18,8), dpi=78, edgecolor='r')
    ax = plt.gca()  
    plt.plot(yTestPred, color = 'blue', label = 'Predicted BTC Price by our Model')
    plt.plot(yActualTest, color = 'red', label = 'Actual BTC Price')
    plt.title('Bitcoin Price Prediction of the Partial Dataset using RNN', fontsize=32)
    x=range(len(ySet) - trainSize)
    labels = np.array(dataSet['Timestamp'])
    labels = labels[trainSize:]
    plt.xticks(np.arange(x[0], x[len(x) - 1], tickNum), labels[np.arange(x[0], x[len(x)-1], tickNum)], rotation = '32')
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(13)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(13)
    plt.xlabel('Time', fontsize=25)
    plt.ylabel('BTC Price', fontsize=25)
    plt.legend(loc=2, prop={'size': 15})
    plt.show()
    print(r2_score(yTest, yTestPred))
    print(max_error(yTest, yTestPred))
    print(explained_variance_score(yTest, yTestPred))
    

if __name__ == "__main__":
    main()
