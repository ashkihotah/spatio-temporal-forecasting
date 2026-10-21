import keras

class WeightedGraphConv(keras.layers.Layer):

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        dim = 2 * dim
        self.dense = keras.layers.Dense(dim, use_bias=False)

    def build(self, input_shape: tuple[keras.KerasTensor, keras.KerasTensor]):
        features_shape = input_shape[0]
        feature_dim = features_shape[-1]
        if feature_dim is None:
            raise ValueError("Feature dimension must be static to build WeightedGraphConv.")

        self.dense.build((None, feature_dim))
        super().build(input_shape)

    def call(self, inputs: tuple[keras.KerasTensor, keras.KerasTensor]) -> keras.KerasTensor:
        x, adj = inputs
        x = keras.ops.matmul(adj, x)
        out = self.dense(x)
        ls, rs = keras.ops.split(out, 2, axis=-1)
        out = ls * keras.ops.sigmoid(rs)
        return out
