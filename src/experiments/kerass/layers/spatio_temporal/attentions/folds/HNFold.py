import keras

class HNFold(keras.layers.Layer):

    def _fold_qkv(self, query, keys, values):
        q_shape = keras.ops.shape(query)
        k_shape = keras.ops.shape(keys)
        v_shape = keras.ops.shape(values)
        B = q_shape[0]
        N = q_shape[1]
        H = k_shape[1]
        query_features = q_shape[2]
        value_features = v_shape[2]

        # Fold Query: (B, N, F) -> (B, N, F)
        # Fold Keys & Values: (B, H, N, F) -> (B, H * N, F)
        keys_folded = keras.ops.reshape((B, H * N, query_features))
        values_folded = keras.ops.reshape((B, H * N, value_features))
        return query, keys_folded, values_folded

    def build(self, input_shape):
        self.N = input_shape[2]
        super().build(input_shape)

    def get_folded_shape(self, input_shape):
        B, L, N, F = input_shape
        return (B, L * N, F)

    def fold(self, x):
        B, L, N, F = keras.ops.shape(x)
        return keras.ops.reshape(
            x, (B, L * N, F)
        )

    def unfold(self, x):
        B, LN, F = keras.ops.shape(x)
        L = LN // self.N
        return keras.ops.reshape(
            x, (B, L, self.N, F)
        )
