import keras

class NormFF(keras.layers.Layer):

    def __init__(self, ff_units: int, **kwargs):
        super().__init__(**kwargs)
        self.norm = keras.layers.LayerNormalization(name="norm")
        self.ff = keras.layers.Dense(
            units=ff_units, name="ff"
        )

    def build(self, input_shape):
        features_shape, _ = input_shape
        self.norm.build(features_shape)
        self.ff.build(features_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        features, adj = inputs
        x = self.norm(features)
        x = self.ff(x)
        return x