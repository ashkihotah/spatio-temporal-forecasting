import keras

class GraphConv(keras.layers.Layer):
    r"""
    A graph convolutional layer (GCN) as presented by
    [Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907).
    **Mode**: single, mixed, batch.
    This layer computes:
    $$
        \Z = \hat \D^{-1/2} \hat \A \hat \D^{-1/2} \X \W + \b
    $$
    where \( \hat \A = \A + \I \) is the adjacency matrix with added self-loops
    and \(\hat\D\) is its degree matrix.
    """

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

        self.kernel = self.add_weight(
            shape=(in_channels, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
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

        output = keras.ops.matmul(features, self.kernel)
        output = keras.ops.matmul(adj, output)

        if self.use_bias:
            output = output + self.bias
            
        if self.activation is not None:
            output = self.activation(output)
            
        return output, keras.ops.zeros((), dtype=keras.backend.floatx())