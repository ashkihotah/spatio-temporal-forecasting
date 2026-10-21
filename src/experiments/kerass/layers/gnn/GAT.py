import keras

class GAT(keras.layers.Layer):
    def __init__(
        self, 
        channels: int, 
        skipconn: bool=False, 
        normalize: bool=True,
        scale: bool=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.skipconn = skipconn
        self.normalize = normalize

        self.Wq = keras.layers.Dense(self.channels)
        self.Wk = keras.layers.Dense(self.channels)
        self.Wv = keras.layers.Dense(self.channels)

        self.softmax = keras.layers.Softmax()
        if self.normalize:
            self.norm = keras.layers.LayerNormalization(scale=scale)

    def build(self, input_shape):
        features_shape = input_shape[0]
        self.Wq.build(features_shape)
        self.Wk.build(features_shape)
        self.Wv.build(features_shape)
        super().build(input_shape)

    def call(self, inputs: tuple[keras.KerasTensor, keras.KerasTensor]) -> keras.KerasTensor:
        x, adj = inputs

        Q = self.Wq(x) # shape: (B, N, F)
        K = self.Wk(x) # shape: (B, N, F)
        V = self.Wv(x) # shape: (B, N, F)
        scores = keras.ops.matmul(
            Q, keras.ops.transpose(K, (0, 2, 1))
        )
        scores = keras.ops.divide(
            scores, keras.ops.sqrt(
                keras.ops.cast(self.channels, dtype=keras.backend.floatx())
            )
        )

        scores = self.softmax(scores)
        # scores = keras.ops.multiply(scores, adj)
        result = keras.ops.matmul(scores, V)

        if self.skipconn:
            result = keras.ops.add(result, x)

        if self.normalize:
            result = self.norm(result)

        return result, keras.ops.zeros((), dtype=keras.backend.floatx())
