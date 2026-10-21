import keras

from ._base import SpatialAttentionBase

def alibi_slope(num_heads):
    return 2.0 ** (-8.0 / float(num_heads))

@keras.saving.register_keras_serializable(package="experiments")
class ALiBiAttention(SpatialAttentionBase):
    """Content attention with the ALiBi relative linear position bias."""

    requires_position_inputs = True

    def __init__(self, num_heads=1, slope=None, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.slope = slope if slope is not None else alibi_slope(num_heads)
        self.scale = None

    def build(self, input_shape):
        if len(input_shape) != 5:
            raise ValueError("ALiBiAttention requires query and key positions.")
        feature_dim = input_shape[0][-1]
        if feature_dim is None:
            raise ValueError("Query feature dimension must be static.")
        self.scale = self.add_weight(
            name="attention_scale",
            shape=(),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, return_attention_scores=False, training=None):
        query, keys, values, query_position, key_positions = self._split_inputs(inputs)
        if query_position is None or key_positions is None:
            raise ValueError("ALiBiAttention requires positions.")
        query_position = keras.ops.squeeze(query_position, axis=-1)
        key_positions = keras.ops.squeeze(key_positions, axis=-1)
        query_folded, keys_folded, values_folded = self._fold_qkv(query, keys, values)
        logits = keras.ops.matmul(
            query_folded,
            keras.ops.transpose(keys_folded, axes=[0, 2, 1]),
        )
        logits = logits * self.scale / keras.ops.sqrt(
            keras.ops.cast(keras.ops.shape(query)[-1], logits.dtype)
        )
        key_positions = keras.ops.transpose(key_positions, axes=[0, 2, 1])
        relative_positions = keras.ops.expand_dims(query_position, axis=-1) - key_positions
        relative_positions = keras.ops.reshape(
            relative_positions,
            (keras.ops.shape(relative_positions)[0] * keras.ops.shape(relative_positions)[1],
             keras.ops.shape(relative_positions)[2]),
        )
        logits = logits - self.slope * keras.ops.expand_dims(relative_positions, axis=1)
        scores = keras.ops.softmax(logits, axis=-1)
        output = keras.ops.matmul(scores, values_folded)
        output = self._unfold_output(output, query)
        if return_attention_scores:
            return output, self._unfold_scores(scores, query, keys)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "slope": self.slope})
        return config
