import keras

def rotate_half(x):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = keras.ops.stack([-odd, even], axis=-1)
    return keras.ops.reshape(rotated, keras.ops.shape(x))

def rotary_embedding(x, positions):
    """Apply RoPE to pairs of channels, leaving one odd channel unchanged."""
    features = keras.ops.shape(x)[-1]
    paired_features = features - features % 2
    paired = x[..., :paired_features]
    remainder = x[..., paired_features:]
    indices = keras.ops.arange(paired_features // 2, dtype="float32")
    inverse_frequency = keras.ops.power(
        10000.0,
        -2.0 * indices / keras.ops.cast(features, "float32"),
    )
    angles = keras.ops.expand_dims(positions, axis=-1) * inverse_frequency
    angles = keras.ops.repeat(angles, repeats=2, axis=-1)
    rotated = paired * keras.ops.cos(angles) + rotate_half(paired) * keras.ops.sin(angles)
    return keras.ops.concatenate([rotated, remainder], axis=-1)

class RoPE(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        features, positions = inputs
        emb = rotary_embedding(features, positions)
        return emb