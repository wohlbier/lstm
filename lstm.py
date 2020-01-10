#!/usr/bin/env python
import numpy as np

if __name__ == "__main__":
    inputs = np.random.random([32, 10, 8]).astype(np.float32)
# lstm = tf.keras.layers.LSTM(4)
# 
# output = lstm(inputs)  # The output has shape `[32, 4]`.
# 
# lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
# 
# # whole_sequence_output has shape `[32, 10, 4]`.
# # final_memory_state and final_carry_state both have shape `[32, 4]`.
# whole_sequence_output, final_memory_state, final_carry_state = lstm(inputs)
