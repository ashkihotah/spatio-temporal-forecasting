import keras

class SpatialAttention(keras.layers.Layer):

    def __init__(
        self, 
        fold_layer: keras.layers.Layer, 
        use_scale: bool = False,
        with_skip_connection: bool = False,
        with_normalization: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fold_layer = fold_layer
        self.with_skip_connection = with_skip_connection
        self.with_normalization = with_normalization
        
        self.attention_layer = keras.layers.Attention(use_scale=use_scale)
        if self.with_normalization:
            self.norm_layer = keras.layers.LayerNormalization()

    def build(self, input_shape):
        queries = input_shape[0]
        keys = input_shape[1]
        values = input_shape[2]

        self.fold_layer.build(queries)

        folded_queries_shape = self.fold_layer.get_folded_shape(queries)
        folded_keys_shape = self.fold_layer.get_folded_shape(keys)
        folded_values_shape = self.fold_layer.get_folded_shape(values)

        self.attention_layer.build([
            folded_queries_shape,
            folded_values_shape,
            folded_keys_shape
        ])
        
        super().build(input_shape)

    def get_attn_scores_shape(self, queries_shape, attn_set_shape):
        folded_queries_shape = self.fold_layer.get_folded_shape(queries_shape)
        folded_attn_set_shape = self.fold_layer.get_folded_shape(attn_set_shape)

        _, Tq, _ = folded_queries_shape
        _, Ta, _ = folded_attn_set_shape

        return self.fold_layer.get_unfolded_shape((None, Tq, Ta))[2:]

    def call(self, inputs, return_attention_scores=False, training=None):
        queries, values, keys = inputs

        folded_queries = self.fold_layer.fold(queries)
        folded_keys = self.fold_layer.fold(keys)
        folded_values = self.fold_layer.fold(values)
        
        output, scores = self.attention_layer(
            [folded_queries, folded_values, folded_keys],
            return_attention_scores=True,
            training=training,
        )

        output = self.fold_layer.unfold(output)

        if self.with_skip_connection:
            output = output + queries

        if self.with_normalization:
            output = self.norm_layer(output)

        if return_attention_scores:
            return output, self.fold_layer.unfold(scores)
        return output