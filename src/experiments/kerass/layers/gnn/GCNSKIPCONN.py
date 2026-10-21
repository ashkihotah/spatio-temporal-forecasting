import keras

class GCNSKIPCONN(keras.layers.Layer):

    def __init__(
        self,
        channels: int,
        activation=None,
        use_bias: bool = True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape: tuple[keras.KerasTensor, keras.KerasTensor]):
        in_channels = input_shape[0][-1]

        self.self_kernel = self.add_weight(
            shape=(in_channels, self.channels),
            initializer=self.kernel_initializer,
            name="self_kernel",
        )
        self.spatial_kernel = self.add_weight(
            shape=(in_channels, self.channels),
            initializer=self.kernel_initializer,
            name="spatial_kernel",
        )
        self.alpha = self.add_weight(
            shape=(1,),
            initializer="ones",
            name="alpha",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name="bias",
            )
        super().build(input_shape)

    def call(
        self, inputs: tuple[keras.KerasTensor, keras.KerasTensor]
    ) -> keras.KerasTensor:
        features, adj = inputs

        self_transformed = keras.ops.matmul(features, self.self_kernel)
        spatial_transformed = keras.ops.matmul(features, self.spatial_kernel)
        spatial_transformed = keras.ops.matmul(adj, spatial_transformed)
        output = self_transformed + self.alpha * spatial_transformed

        if self.use_bias:
            output = output + self.bias
            
        if self.activation is not None:
            output = self.activation(output)
            
        return output, keras.ops.zeros((), dtype=keras.backend.floatx())