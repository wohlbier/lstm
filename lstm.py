#!/usr/bin/env python
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    # data presented with 
    batch = 32       # number of samples presented to network at a time
    timesteps = 10   # sequence length
    feature_size = 8 # feature vector per sample per sequence element

    # [batch, timesteps, feature]
    inputs = np.random.random([32, 10, 8]).astype(np.float32)

    # units is number of internal units
    units = 4
    lstm = tf.keras.layers.LSTM(units = units)
    output = lstm(inputs)  # The output has shape `[32, 4]`.

    lstm.summary()

#    lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
#
#    # whole_sequence_output has shape `[32, 10, 4]`.
#    # final_memory_state and final_carry_state both have shape `[32, 4]`.
#    whole_sequence_output, final_memory_state, final_carry_state = lstm(inputs)
#
    print(inputs)
    print(inputs.shape)
#    print(whole_sequence_output)
#    print(whole_sequency_output.shape)
#    print(final_memory_state)
#    print(final_memory_state.shape)
#    print(final_carry_state)
#    print(final_carry_state.shape)
