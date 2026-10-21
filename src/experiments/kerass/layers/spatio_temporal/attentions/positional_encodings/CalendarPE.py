import keras

class CalendarPE(keras.layers.Layer):

    def __init__(self, states_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.query_time_projection = keras.layers.Dense(
            states_dim, name="query_time_projection"
        )
        self.key_time_projection = keras.layers.Dense(
            states_dim, name="key_time_projection"
        )

    def build(self, input_shape):
        q_shape, k_shape, v_shape, q_time_shape, k_time_shape = input_shape
        self.query_time_projection.build(q_time_shape)
        self.key_time_projection.build(k_time_shape)
        self.attention_layer.build(
            self.fold_layer.get_folded_shape(
                (q_shape, k_shape, v_shape)
            )
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        query, keys, values, query_time, key_times = inputs

        query_time = self.query_time_projection(query_time)
        key_times = self.key_time_projection(key_times)

        query = query + query_time
        keys = keys + key_times

        return query, keys, values