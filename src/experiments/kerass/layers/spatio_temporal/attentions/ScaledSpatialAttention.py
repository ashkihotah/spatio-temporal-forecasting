import keras

class ScaledSpatialAttention(keras.layers.Layer):
    
    def __init__(
        self, 
        fold_layer: keras.layers.Layer, 
        pe_layer: keras.layers.Layer = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fold_layer = fold_layer
        self.pe_layer = pe_layer

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="attention_scale",
            shape=(),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, return_attention_scores=False, training=None):
        query, keys, values = inputs

        query_folded, keys_folded, values_folded = self._fold_qkv(
            query, keys, values
        )
        transposed_keys_folded = keras.ops.transpose(
            keys_folded, axes=[0, 2, 1]
        )
        
        logits = keras.ops.matmul(
            query_folded, transposed_keys_folded
        )
        logits = logits * self.scale / keras.ops.sqrt(
            keras.ops.cast(keras.ops.shape(query_folded)[-1], logits.dtype)
        )
        scores = keras.ops.softmax(logits, axis=-1)
        output = keras.ops.matmul(scores, values_folded)
        
        output = self._unfold_output(output, query)
        if return_attention_scores:
            return output, self._unfold_scores(scores, query, keys)
        return output
