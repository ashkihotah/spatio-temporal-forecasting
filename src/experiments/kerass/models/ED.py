import keras

from experiments.kerass.layers.sequences.constants import (
    DECODER_LOGS_KEY, DECODER_OUTPUTS_KEY, 
    ENCODER_LOGS_KEY, ENCODER_OUTPUTS_KEY,
    LOGS_KEY, OUTPUTS_KEY, STATE_KEY
)
from experiments.kerass.layers.sequences.arnn import AutoRegressiveRNN
from experiments.kerass.layers.sequences.rnn import RNN

class ED(keras.Model):

    def __init__(
        self,
        p: int,

        encoder_cell: keras.layers.Layer,
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

        self.encoder = RNN(
            cell=encoder_cell, 
            return_outputs=return_encoder_cell_outputs,
            return_logs=return_encoder_cell_logs,
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
        self.decoder.build((
            decoder_initial_outputs_shape,
            decoder_inputs_shape
        ))

        super().build(input_shape)

    def call(self, inputs, training=None):

        encoder_inputs = inputs[0]
        decoder_inputs = inputs[1]
        decoder_initial_outputs = inputs[2]

        enc_out = self.encoder(encoder_inputs, training=training)
        encoder_final_state = enc_out[STATE_KEY] # (B, H, ...)

        if self.use_encoder_last_outputs_as_decoder_initial_outputs:
            enc_preds = enc_out[OUTPUTS_KEY] # (B, H, ...)
            decoder_initial_outputs = enc_preds[:, -1, ...] # (B, N, ...)

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
