import keras

def sinusoidal_encoding(positions, features: int):
    """Return the original Transformer sinusoidal encoding."""
    positions = keras.ops.cast(positions, "float32")
    half_features = (features + 1) // 2
    indices = keras.ops.arange(half_features, dtype="float32")
    frequencies = keras.ops.power(
        10000.0,
        -2.0 * indices / keras.ops.cast(features, "float32"),
    )
    angles = keras.ops.expand_dims(positions, axis=-1) * frequencies
    encoding = keras.ops.stack(
        [keras.ops.sin(angles), keras.ops.cos(angles)], axis=-1
    )
    encoding = keras.ops.reshape(
        encoding,
        (*keras.ops.shape(positions), half_features * 2),
    )
    return encoding[..., :features]

class SinusoidalPE(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        features, positions = inputs
        pe = sinusoidal_encoding(
            positions, keras.ops.shape(features)[-1]
        )
        return features + pe