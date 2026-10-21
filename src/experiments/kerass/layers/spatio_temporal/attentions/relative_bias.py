import keras

class RelativeTimeBiasAttention(keras.layers.Layer):

    def __init__(self, bias_units=32, **kwargs):
        super().__init__(**kwargs)
        self.bias_units = bias_units
        self.bias_network = keras.Sequential([
            keras.layers.Dense(bias_units, activation="tanh"),
            keras.layers.Dense(1),
        ], name="relative_time_bias")
        self.scale = None

    def build(self, input_shape):
        if len(input_shape) != 5:
            raise ValueError(
                "RelativeTimeBiasAttention requires query and key time inputs."
            )
        query_shape, keys_shape, values_shape, query_time_shape, key_time_shape = input_shape
        feature_dim = query_shape[-1]
        time_dim = query_time_shape[-1]
        if feature_dim is None or time_dim is None:
            raise ValueError("Content and temporal dimensions must be static.")
        self.bias_network.build((None, None, time_dim))
        self.scale = self.add_weight(
            name="attention_scale",
            shape=(),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, return_attention_scores=False, training=None):
        query, keys, values, query_time, key_times = self._split_inputs(inputs)
        if query_time is None or key_times is None:
            raise ValueError("RelativeTimeBiasAttention requires temporal inputs.")

        query_folded, keys_folded, values_folded = self._fold_qkv(
            query, keys, values
        )
        content_logits = keras.ops.matmul(
            query_folded,
            keras.ops.transpose(keys_folded, axes=[0, 2, 1]),
        )
        content_logits = content_logits * self.scale / keras.ops.sqrt(
            keras.ops.cast(keras.ops.shape(query)[-1], content_logits.dtype)
        )

        key_times = keras.ops.transpose(key_times, axes=[0, 2, 1, 3])
        relative_time = keras.ops.expand_dims(query_time, axis=2) - key_times
        relative_time = keras.ops.reshape(
            relative_time,
            (
                keras.ops.shape(relative_time)[0] * keras.ops.shape(relative_time)[1],
                keras.ops.shape(relative_time)[2],
                keras.ops.shape(relative_time)[3],
            ),
        )
        temporal_bias = self.bias_network(relative_time, training=training)
        temporal_bias = keras.ops.squeeze(temporal_bias, axis=-1)
        temporal_bias = keras.ops.expand_dims(temporal_bias, axis=1)
        logits = content_logits + temporal_bias
        scores_folded = keras.ops.softmax(logits, axis=-1)
        output_folded = keras.ops.matmul(scores_folded, values_folded)
        output = self._unfold_output(output_folded, query)
        if return_attention_scores:
            return output, self._unfold_scores(scores_folded, query, keys)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"bias_units": self.bias_units})
        return config
