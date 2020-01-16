#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

if __name__ == "__main__":

    # data specs
    batch = 32       # number of samples presented to network at a time
    timesteps = 10   # sequence length
    feat_sz = 8      # feature vector per sample per sequence element
    input = Input(shape=(timesteps,feat_sz))

    # number of internal LSTM units
    units = 4
    lstm = LSTM(units=units, \
                return_sequences=True, return_state=True,)(input)
    model = Model(inputs=input, outputs=lstm)

    print(model.summary())
    plot_model(model, to_file='lstm.png')

    # [batch, timesteps, feature]
    inputs = np.random.random([batch, timesteps, feat_sz]).astype(np.float32)

    # whole_sequence_output has shape `[32, 10, 4]`.
    # final_memory_state and final_carry_state both have shape `[32, 4]`.
    whole_sequence_output, final_memory_state, final_carry_state \
        = model(inputs)
