import keras

from experiments.kerass.layers.sequences.constants import (
    DECODER_OUTPUTS_KEY, ENCODER_OUTPUTS_KEY
)

class STEDConvolution(keras.layers.Layer):

    def __init__(self, p: int, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.conv = keras.layers.Conv2D(
            filters=self.p, 
            kernel_size=(2, 2), 
            data_format="channels_last", 
            padding="same"
        )
        self.head = keras.layers.Dense(
            units=output_dim,
            name="output_projection"
        )

    def build(self, input_shape):
        encoder_outputs_shape = input_shape.pop(ENCODER_OUTPUTS_KEY)
        self.head.build(encoder_outputs_shape)
        B, H, N, O = encoder_outputs_shape
        conv_input_shape = (B, N, O, 1)
        self.conv.build(conv_input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):

        dec_outputs = inputs.pop(DECODER_OUTPUTS_KEY) # (B, P, N, O)
        enc_outputs = inputs.pop(ENCODER_OUTPUTS_KEY) # (B, H, N, O)

        encoder_final_h = enc_outputs[:, -1, :, :] # (B, N, O)
        # (B, N, O, 1)
        conv_input = keras.ops.expand_dims(encoder_final_h, axis=-1)
        # (B, N, O, P)
        conv_output = self.conv(conv_input)
        # (B, P, N, O)
        conv_output = keras.ops.transpose(conv_output, (0, 3, 1, 2))
        # (B, P, N, O) + (B, P, N, O)
        dec_outputs = dec_outputs + conv_output

        enc_outputs = self.head(enc_outputs)
        dec_outputs = self.head(dec_outputs)

        return {
            **inputs,
            DECODER_OUTPUTS_KEY: dec_outputs,
            ENCODER_OUTPUTS_KEY: enc_outputs
        }