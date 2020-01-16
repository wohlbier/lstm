#!/usr/bin/env python
import numpy as np
import pandas as pd
import sklearn.model_selection as sk
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

def split_sequence(sequence,n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

if __name__ == "__main__":

    print("Build model")

    # data specs
    batch_sz = 32 # number of samples presented to network at a time
    timesteps = 5 # sequence length
    n_feats = 1   # feature vector per sample per sequence element
    input = Input(shape=(timesteps,n_feats))

    # number of internal LSTM units
    units = 4
    lstm = LSTM(units=units,activation='relu')(input)
    dense = Dense(n_feats)(lstm)
    model = Model(inputs=input, outputs=dense)

    print(model.summary())
    plot_model(model, to_file='lstm.png')

    # compile the model
    model.compile(optimizer='adam',loss='mse')

    print("Read and process data")

    # data prep
    df = pd.read_csv("MU.csv",usecols=["Date","Close","Volume"])
    close = df["Close"]
    X, y = split_sequence(close,timesteps)
    # reshape for one feature
    X = X.reshape(X.shape[0], X.shape[1], n_feats)

    X_train, X_test, y_train, y_test = \
       sk.train_test_split(X,y,test_size=0.2,random_state=42)

    print("Fit the model")

    # fit the model
    model.fit(X_train,y_train,batch_size=1,epochs=5,verbose=1)

    print("Evaluate the model")

    # evaluate the model
    results = model.evaluate(X_test,y_test,verbose=0)
    print('test loss, test acc:', results)
