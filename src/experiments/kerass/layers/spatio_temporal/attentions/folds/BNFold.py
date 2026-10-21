import keras

class BNFold(keras.layers.Layer):

    def build(self, input_shape):
        self.N = input_shape[2]
        super().build(input_shape)

    def get_folded_shape(self, input_shape):
        B, L, N, F = input_shape
        return (B, L, F) # * N

    def get_unfolded_shape(self, folded_shape):
        B, L, F = folded_shape
        return (B, L, self.N, F)

    def fold(self, x):
        B, L, N, F = keras.ops.shape(x)
        return keras.ops.reshape(
            keras.ops.transpose(x, axes=[0, 2, 1, 3]), # [B, N, L, F]
            (B * N, L, F), # [B * N, L, F]
        )

    def unfold(self, x):
        _, L, F = keras.ops.shape(x)
        return keras.ops.transpose(
            keras.ops.reshape(x, (-1, self.N, L, F)), # [B, N, L, F]
            axes=[0, 2, 1, 3], # [B, L, N, F]
        )
