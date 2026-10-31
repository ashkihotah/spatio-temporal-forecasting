import keras
from typing import Tuple


class LearnableVirtualProbesPooling(keras.layers.Layer):
    """Learnable Virtual Probes Graph Compressor (Set/Cross-Attention Pooling).

    Uses M trainable latent probe vectors to query node representations.
    Incorporates graph degree centrality into keys for topology awareness.
    Returns a unified compressed embedding R^{B x D}.
    """

    def __init__(
        self,
        output_dim: int,
        num_probes: int = 4,
        probe_dim: int = 32,
        use_degree_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_probes = num_probes
        self.probe_dim = probe_dim
        self.use_degree_bias = use_degree_bias

        self.k_proj = keras.layers.Dense(self.probe_dim, use_bias=False, name="k_proj")
        self.v_proj = keras.layers.Dense(self.probe_dim, use_bias=False, name="v_proj")
        
        if self.use_degree_bias:
            self.deg_proj = keras.layers.Dense(
                self.probe_dim, use_bias=False, name="deg_proj"
            )

        self.out_proj = keras.layers.Dense(self.output_dim, name="probe_projection")

    def build(self, input_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]):
        x_shape, a_shape = input_shape
        feature_dim = x_shape[-1]

        # 1. Trainable Latent Probes: (M, probe_dim)
        self.virtual_probes = self.add_weight(
            name="virtual_probes",
            shape=(self.num_probes, self.probe_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # 2. Build sublayers
        self.k_proj.build((None, feature_dim))
        self.v_proj.build((None, feature_dim))

        if self.use_degree_bias:
            self.deg_proj.build((None, 1))

        # Flattened probe outputs: M * probe_dim
        self.out_proj.build((None, self.num_probes * self.probe_dim))

        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
        training: bool = False,
    ) -> keras.KerasTensor:
        x, a = inputs  # x: (B, N, F), a: (B, N, N)
        batch_size = keras.ops.shape(x)[0]
        scale = 1.0 / keras.ops.sqrt(
            keras.ops.cast(self.probe_dim, dtype=keras.ops.dtype(x))
        )

        # 1. Project Keys and Values from nodes
        keys = self.k_proj(x)    # (B, N, probe_dim)
        values = self.v_proj(x)  # (B, N, probe_dim)

        # 2. Inject structural degree bias into Keys
        if self.use_degree_bias:
            degrees = keras.ops.sum(a, axis=-1, keepdims=True)  # (B, N, 1)
            deg_features = self.deg_proj(degrees)              # (B, N, probe_dim)
            keys = keys + deg_features

        # 3. Broadcast Probes across batch: (B, M, probe_dim)
        probes = keras.ops.broadcast_to(
            keras.ops.expand_dims(self.virtual_probes, axis=0),
            (batch_size, self.num_probes, self.probe_dim),
        )

        # 4. Cross-Attention: Probes attend over Nodes
        # (B, M, probe_dim) x (B, probe_dim, N) -> (B, M, N)
        scores = keras.ops.matmul(probes, keras.ops.transpose(keys, axes=(0, 2, 1)))
        scores = scores * scale
        attn_weights = keras.ops.softmax(scores, axis=-1)  # (B, M, N)

        # 5. Aggregate Values: (B, M, N) x (B, N, probe_dim) -> (B, M, probe_dim)
        probe_embeddings = keras.ops.matmul(attn_weights, values)

        # 6. Flatten probe dimension: (B, M * probe_dim)
        flattened = keras.ops.reshape(
            probe_embeddings, (batch_size, self.num_probes * self.probe_dim)
        )

        # 7. Final Projection: (B, D)
        return self.out_proj(flattened)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "num_probes": self.num_probes,
                "probe_dim": self.probe_dim,
                "use_degree_bias": self.use_degree_bias,
            }
        )
        return config