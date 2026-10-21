import keras

from experiments.kerass.layers.sequences.constants import DECODER_OUTPUTS_KEY, ENCODER_OUTPUTS_KEY

class STReshapedHead(keras.layers.Layer):

    def __init__(self, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.head_layer = keras.layers.Dense(
            units=output_dim,
            name="output_projection"
        )

    def build(self, input_shape):
        dec_outputs_shape = input_shape.pop(DECODER_OUTPUTS_KEY) # (B, P, N, S)

        B, P, N, S = dec_outputs_shape
        dec_outputs_shape = (B, P, N * S)
        
        self.head_layer.build(dec_outputs_shape)
        super().build(dec_outputs_shape)

    def _apply_head(self, inputs):
        B, P, N, S = keras.ops.shape(inputs)
        inputs = keras.ops.reshape(
            inputs, (B, P, N * S)
        )

        outputs = self.head_layer(inputs)
        outputs = keras.ops.expand_dims(outputs, axis=-1) # ( B, P, N, 1 )
        return outputs

    def call(self, inputs, training=None):
        dec_outputs = inputs.pop(DECODER_OUTPUTS_KEY) # (B, P, ...)
        enc_outputs = inputs.pop(ENCODER_OUTPUTS_KEY)

        dec_outputs = self._apply_head(dec_outputs)
        enc_outputs = self._apply_head(enc_outputs)
        
        return {
            **inputs,
            ENCODER_OUTPUTS_KEY: enc_outputs,
            DECODER_OUTPUTS_KEY: dec_outputs
        }