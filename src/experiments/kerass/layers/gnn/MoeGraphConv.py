import keras
from experiments.kerass.layers.routers.MoeRouter import MoeRouter

class MoeGraphConv(keras.layers.Layer):
    r"""
    A Graph Convolutional Layer (GCN) as presented by
    [Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907).
    
    Supports both standard single-kernel execution and Layer-Level
    Mixture-of-Experts (MoE) weight blending.
    """

    def __init__(
        self,
        channels: int,

        num_experts: int = None,
        top_k: int = 1,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,

        activation=None, # "relu",
        use_bias: bool = True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",

        alpha_initializer="ones",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        if isinstance(alpha_initializer, (int, float)):
            self.alpha_initializer = keras.initializers.Constant(alpha_initializer)
        else:
            self.alpha_initializer = alpha_initializer

        self.moe_router = MoeRouter(
            n_experts=self.num_experts,
            top_k=self.top_k,
            aux_loss_weight=self.aux_loss_weight,
            z_loss_weight=self.z_loss_weight,
        )

    def build(self, input_shape: tuple[keras.KerasTensor, keras.KerasTensor]):
        features_shape = input_shape[0]
        in_channels = features_shape[-1]

        self.self_kernel = self.add_weight(
            shape=(self.num_experts, in_channels, self.channels),
            initializer=self.kernel_initializer,
            name="self_kernel",
        )
        self.spatial_kernels = self.add_weight(
            shape=(self.num_experts, in_channels, self.channels),
            initializer=self.kernel_initializer,
            name="spatial_kernels",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.num_experts, self.channels),
                initializer=self.bias_initializer,
                name="bias",
            )
        self.moe_router.build(features_shape)

        self.alpha = self.add_weight(
            shape=(1,),
            initializer=self.alpha_initializer,
            name="alpha",
        )

        super().build(input_shape)

    def call(
        self, 
        inputs: tuple[keras.KerasTensor, keras.KerasTensor],
        training: bool = None,
    ):
        features, adj = inputs
        # features: (B, N, F)
        # adj: (B, N, N)

        routing_weights, router_loss = self.moe_router(
            features, training=training
        ) # routing_weights: (B, N, E)

        self_transformed = keras.ops.einsum(
            "b n i, e i o -> b n e o", features, self.self_kernel
        )
        spatial_transformed = keras.ops.einsum(
            "b n i, b i j -> b n j", adj, features
        )
        spatial_transformed = keras.ops.einsum(
            "b n i, e i o -> b n e o", spatial_transformed, self.spatial_kernels
        )

        expert_outputs = self_transformed + self.alpha * spatial_transformed

        if self.use_bias:
            expert_outputs = expert_outputs + self.bias

        output = keras.ops.einsum(
            "b n e o, b n e -> b n o", expert_outputs, routing_weights
        ) # output: (B, N, channels)

        if self.activation is not None:
            output = self.activation(output)

        return output, router_loss