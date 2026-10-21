from typing import TypedDict

import keras
import optree

from experiments.kerass.layers.sequences.rnn import RNN
from experiments.kerass.layers.sequences.arnn import AutoRegressiveRNN
from experiments.kerass.layers.wrappers import CellWithAttention
from experiments.kerass.utils import is_node_a_tensor_shape
from experiments.kerass.layers.sequences.constants import (
    DECODER_LOGS_KEY, DECODER_OUTPUTS_KEY, 
    ENCODER_LOGS_KEY, ENCODER_OUTPUTS_KEY,
    LOGS_KEY, OUTPUTS_KEY, STATE_KEY
)

def get_keys_and_values(
    state: keras.KerasTensor,
    states_mask: any
):    

    flat_state = optree.tree_leaves(state)
    flat_mask = keras.tree.flatten(states_mask)

    states = []
    iterator = zip(flat_state, flat_mask)
    for state, wanted in iterator:
        if wanted:
            shape = keras.ops.shape(state)
            B = shape[0]
            H = shape[1]
    
            # state: (B, H, ...)
            S = keras.ops.expand_dims(state, axis=1) # (B, 1, H, ...)
            S = keras.ops.broadcast_to(S, (B, H, H, *shape[2:])) # (B, H, H, ...)
            states.append(S)

            # TODO: dovrebbe essere B P H non B H H

    return states, states

def get_key_times(
    encoder_features: keras.KerasTensor,
    temporal_feature_indices: list[int],
):
    temporal_features = keras.ops.take(
        encoder_features[..., 1:],
        keras.ops.convert_to_tensor(temporal_feature_indices, dtype="int32"),
        axis=-1,
    )
    B, H, *other_shape = keras.ops.shape(temporal_features)
    temporal_features = keras.ops.expand_dims(temporal_features, axis=1)
    # TODO: dovrebbe essere B P H non B H H
    return keras.ops.broadcast_to(temporal_features,(B, H, H, *other_shape))

def get_key_positions(keys: keras.KerasTensor):
    shape = keras.ops.shape(keys)
    B, H, *other_shape = shape
    other_1_shape = (1,) * (len(other_shape) - 1)

    positions = keras.ops.arange(H, dtype="float32")
    positions = keras.ops.reshape(positions, (1, H, *other_1_shape, 1))
    positions = keras.ops.broadcast_to(positions, (B, H, *other_1_shape, 1))

    positions = keras.ops.expand_dims(positions, axis=1)
    # TODO: dovrebbe essere B P H non B H H
    return keras.ops.broadcast_to(positions, (B, H, H, *other_1_shape, 1))

def get_decoder_positions(
    decoder_cell_state_size: tuple[int], 
    encoder_steps: int
):
    shape = (None,) + keras.ops.shape(decoder_cell_state_size)
    B, P, *other_shape = shape
    other_1_shape = (1,) * (len(other_shape) - 1)
    positions = keras.ops.arange(
        encoder_steps, encoder_steps + P,  dtype="float32",
    )
    positions = keras.ops.reshape(positions, (1, P, *other_1_shape, 1))
    return keras.ops.broadcast_to(positions, (B, P, *other_1_shape, 1))

class AttentionConfig(TypedDict):
    attention_layers: list[keras.layers.Layer]
    states_mask: any

class STAED(keras.Model):

    def __init__(
        self,
        p: int,

        encoder_cell: keras.layers.Layer,
        attention_config: AttentionConfig,
        decoder_cell: keras.layers.Layer = None,

        use_encoder_last_outputs_as_decoder_initial_outputs: bool = False,

        return_encoder_cell_outputs: bool = False,
        return_decoder_cell_outputs: bool = False,

        return_encoder_cell_logs: bool = False,
        return_decoder_cell_logs: bool = False,

        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.return_encoder_cell_outputs = return_encoder_cell_outputs
        self.return_decoder_cell_outputs = return_decoder_cell_outputs
        self.return_encoder_cell_logs = return_encoder_cell_logs
        self.return_decoder_cell_logs = return_decoder_cell_logs
        self.use_encoder_last_outputs_as_decoder_initial_outputs = use_encoder_last_outputs_as_decoder_initial_outputs
        
        if decoder_cell is None:
            decoder_cell = encoder_cell

        return_encoder_cell_outputs = return_encoder_cell_outputs or use_encoder_last_outputs_as_decoder_initial_outputs

        self.attention_layers = attention_config["attention_layers"]
        self.states_mask = attention_config["states_mask"]
        self.temporal_feature_indices = attention_config.pop(
            "temporal_feature_indices", None
        )
        decoder_cell = CellWithAttention(
            cell=decoder_cell,
            **attention_config,
            name="decoder_cell_with_attention"
        )
        self.use_positional_attention = any(
            getattr(layer, "requires_position_inputs", False)
            for layer in self.attention_layers
        )

        self.encoder = RNN(
            cell=encoder_cell, 
            return_outputs=return_encoder_cell_outputs,
            return_logs=True,
            name="encoder"
        )
        self.decoder = AutoRegressiveRNN(
            cell=decoder_cell,
            prediction_horizon=p,
            return_logs=return_decoder_cell_logs,
            name="decoder"
        )

    def build(self, input_shape):
        encoder_input_shape = input_shape[0]
        decoder_inputs_shape = input_shape[1]
        decoder_initial_outputs_shape = input_shape[2]

        self.encoder.build(encoder_input_shape)

        flat_encoder_input_shape = optree.tree_leaves(
            encoder_input_shape, is_leaf=is_node_a_tensor_shape
        )
        B = flat_encoder_input_shape[0][0]
        H = flat_encoder_input_shape[0][1]
        N = flat_encoder_input_shape[0][2]

        self.encoder.cell.return_state = True

        log_size = self.encoder.cell.log_size
        if STATE_KEY not in log_size:
            raise ValueError(
                f"The encoder cell must return its state in the logs "
                f"under the key '{STATE_KEY}' when using attention. "
                f"Please set 'return_state=True' when instantiating the encoder cell."
            )

        enc_state_shape = self.encoder.cell.state_size

        state_layout_schema = optree.tree_leaves(
            enc_state_shape, 
            is_leaf=is_node_a_tensor_shape
        )
        state_mask_flat = keras.tree.flatten(self.states_mask)

        keys_shapes = []
        values_shapes = []
        iterator = zip(state_layout_schema, state_mask_flat)
        for i, (state_shape, wanted) in enumerate(iterator):
            if wanted:
                keys_shape = (B, H, H, *state_shape)
                values_shape = (B, H, H, *state_shape)

                keys_shapes.append(keys_shape)
                values_shapes.append(values_shape)

        decoder_inputs_shape = (
            decoder_inputs_shape,
            keys_shapes,
            values_shapes
        )
        decoder_feature_shape = decoder_inputs_shape[0][0]
        P = decoder_feature_shape[1]
        if self.temporal_feature_indices is not None:
            temporal_dim = len(self.temporal_feature_indices)
            query_time_shape = (B, P, N, temporal_dim)
            key_time_shape = (
                B, H, H, N, temporal_dim
            )
            decoder_inputs_shape = (
                *decoder_inputs_shape,
                query_time_shape,
                key_time_shape,
            )
        if self.use_positional_attention:
            query_position_shape = (B, P, N, 1)
            key_position_shape = (B, H, H, N, 1)
            decoder_inputs_shape = (
                *decoder_inputs_shape,
                query_position_shape,
                key_position_shape,
            )

        self.decoder.build((
            decoder_initial_outputs_shape,
            decoder_inputs_shape
        ))

        self.built = True

    def call(self, inputs, training=None):

        encoder_inputs = inputs[0]
        decoder_inputs = inputs[1]
        decoder_initial_outputs = inputs[2]

        enc_out = self.encoder(encoder_inputs, training=training)
        encoder_final_state = enc_out[STATE_KEY] # (B, H, ...)

        if self.use_encoder_last_outputs_as_decoder_initial_outputs:
            enc_preds = enc_out[OUTPUTS_KEY] # (B, H, ...)
            decoder_initial_outputs = enc_preds[:, -1, ...] # (B, N, ...)
        
        enc_h = enc_out[LOGS_KEY][STATE_KEY] # (B, H, ...)
        raw_decoder_inputs = decoder_inputs
        decoder_inputs = (
            decoder_inputs, *get_keys_and_values(
                state=enc_h,
                states_mask=self.states_mask
            )
        )
        
        if self.temporal_feature_indices is not None:
            encoder_features = encoder_inputs[0] # TODO: not agnostic
            decoder_features = raw_decoder_inputs[0] # TODO: not agnostic
            decoder_inputs = (
                *decoder_inputs,
                keras.ops.take(
                    decoder_features,
                    keras.ops.convert_to_tensor(
                        self.temporal_feature_indices, dtype="int32"
                    ),
                    axis=-1,
                ),
                get_key_times(
                    encoder_features, self.temporal_feature_indices
                ),
            )

        if self.use_positional_attention:
            # encoder_features = encoder_inputs[0] # TODO: not agnostic
            decoder_inputs = (
                *decoder_inputs,
                get_decoder_positions(
                    raw_decoder_inputs[0], # TODO: encoder state
                    keras.ops.shape(encoder_features)[1],
                ),
                get_key_positions(enc_h),
            )

        dec_out = self.decoder(
            initial_outputs=decoder_initial_outputs,
            initial_states=encoder_final_state,
            inputs=decoder_inputs,
            training=training
        )

        output_dict = {
            DECODER_OUTPUTS_KEY: dec_out[OUTPUTS_KEY]
        }
        if self.return_encoder_cell_outputs:
            output_dict[ENCODER_OUTPUTS_KEY] = enc_out[OUTPUTS_KEY]
        if self.return_decoder_cell_logs:
            output_dict[DECODER_LOGS_KEY] = dec_out[LOGS_KEY]
        if self.return_encoder_cell_logs:
            output_dict[ENCODER_LOGS_KEY] = enc_out[LOGS_KEY]

        return output_dict
