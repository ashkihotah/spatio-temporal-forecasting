import tensorflow as tf
import keras

from experiments.models import DEC_STATES_KEY, ENC_STATES_KEY, Y_PRED_KEY, ATT_WEIGHTS_KEY

class AutoRegressiveRNN(tf.keras.layers.Layer):

    def __init__(
        self, 
        units: int, p: int,
        cell_type: str = "LSTM", 
        has_attention: bool = False,
        is_bidirectional: bool = False,
        return_hidden_states: bool = False, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.p = p
        self.cell_type = cell_type
        self.has_attention = has_attention
        self.is_bidirectional = is_bidirectional
        self.return_hidden_states = return_hidden_states
        
        if self.cell_type == "LSTM":
            self.cell = tf.keras.layers.LSTMCell(units=self.units)
        elif self.cell_type == "GRU":
            self.cell = tf.keras.layers.GRUCell(units=self.units)

        if self.has_attention:
            self.att = tf.keras.layers.Dense(self.units)

        self.encoder = tf.keras.layers.RNN(
            self.cell, 
            return_sequences=True, 
            return_state=True
        )
    
    def build(self, input_shape):
        # input_shape: (Batch, History, Features) or (Batch, History, Nodes * Features)
        self.feature_dim = input_shape[-1]
        
        # Projections layer maps the hidden state back to the physical feature dimension
        self.projection_layer = tf.keras.layers.Dense(self.feature_dim)
        self.projection_layer.build((input_shape[0], self.units))

        if self.has_attention:
            self.att.build((input_shape[0], self.units))
            
        self.encoder.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        if self.is_bidirectional:
            B = tf.shape(inputs)[0]
            H = tf.shape(inputs)[1]
            reversed_seq = tf.reverse_sequence(
                inputs, tf.fill([B], H), seq_axis=1, batch_axis=0
            )
            inputs = tf.concat([inputs, reversed_seq], axis=1)

        if self.cell_type == "LSTM":
            encoder_states, state_h, state_c = self.encoder(inputs)
            current_carry = [state_h, state_c]
        else: # GRU
            encoder_states, state_h = self.encoder(inputs)
            current_carry = [state_h]

        preds_array = tf.TensorArray(dtype=tf.float32, size=self.p)
        decoder_states_array = tf.TensorArray(dtype=tf.float32, size=self.p)
        
        if self.has_attention:
            att_weights_array = tf.TensorArray(dtype=tf.float32, size=self.p)
        else:
            att_weights_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=False)
            
        i = tf.constant(0)
        
        # REFACTOR: The initial input to the decoder loop is now a projection of the final 
        # encoder state into the physical feature space, rather than raw state_h.
        current_input = self.projection_layer(state_h)

        def condition(i, current_input, current_carry, preds_array, decoder_states_array, att_weights_array):
            return i < self.p

        def body(i, current_input, current_carry, preds_array, decoder_states_array, att_weights_array):
            if self.has_attention:
                memory = tf.expand_dims(current_input, 1)  # [B, 1, feature_dim]
                # Map memory temporarily to units space to match attention dimension
                memory_projected = self.att(memory)
                scores = tf.matmul(
                    memory_projected, tf.transpose(encoder_states, perm=[0, 2, 1])
                )
                weights = tf.nn.softmax(scores)  # [B, 1, H]
                context = tf.matmul(weights, encoder_states)
                
                # Squeeze context and merge with input context if needed
                # For basic attention we can pass the squeezed context into the cell
                current_input_processed = tf.squeeze(context, axis=1)  # [B, units]
                att_weights_array = att_weights_array.write(i, tf.squeeze(weights, axis=1))
            else:
                # If no attention, pass the physical feature vector directly
                current_input_processed = current_input

            # Step the RNN Cell
            next_state, next_carry = self.cell(current_input_processed, current_carry)
            
            # REFACTOR: Project the hidden state output to the physical output space 
            # (e.g., predicting the variables for the next step)
            predicted_output = self.projection_layer(next_state)
            
            # Store predictions and states
            preds_array = preds_array.write(i, predicted_output)
            decoder_states_array = decoder_states_array.write(i, next_state)
            
            # REFACTOR: The predicted output vector becomes the next step's input feature vector
            return i + 1, predicted_output, next_carry, preds_array, decoder_states_array, att_weights_array

        _, _, _, preds_array, decoder_states_array, att_weights_array = tf.while_loop(
            condition, body, [i, current_input, current_carry, preds_array, decoder_states_array, att_weights_array]
        )
        
        preds = preds_array.stack()
        res = tf.transpose(preds, perm=[1, 0, 2]) # Shape: (batch_size, p, feature_dim)

        if self.return_hidden_states:
            res_dict = {Y_PRED_KEY: res}
            res_dict[ENC_STATES_KEY] = encoder_states
            stacked_decoder = decoder_states_array.stack()
            res_dict[DEC_STATES_KEY] = tf.transpose(stacked_decoder, perm=[1, 0, 2])
            if self.has_attention:
                stacked_weights = att_weights_array.stack()
                res_dict[ATT_WEIGHTS_KEY] = tf.transpose(stacked_weights, perm=[1, 0, 2])
            return res_dict
        else:
            return res

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "p": self.p,
            "cell_type": self.cell_type,
            "has_attention": self.has_attention,
            "is_bidirectional": self.is_bidirectional,
            "return_hidden_states": self.return_hidden_states
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)