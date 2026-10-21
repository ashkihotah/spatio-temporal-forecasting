import keras

class MoeRouter(keras.layers.Layer):

    def __init__(
        self,
        n_experts: int,
        top_k: int = 1,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_experts = n_experts
        self.k = min(top_k, self.num_experts)
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight

        # Gating networks for Noisy Top-k Routing
        self.w_gate = keras.layers.Dense(
            self.num_experts, use_bias=False, name="router_gate"
        )
        self.w_noise = keras.layers.Dense(
            self.num_experts, use_bias=False, name="router_noise"
        )

    def build(self, input_shape):
        self.w_gate.build(input_shape)
        self.w_noise.build(input_shape)

        super().build(input_shape)

    def call(
        self, inputs: keras.KerasTensor, training: bool = False
    ) -> tuple[keras.KerasTensor, keras.KerasTensor]:

        router_x = inputs # (batch, ..., router_in_dim)
        clean_logits = self.w_gate(router_x) # (batch, ..., num_experts)

        # (batch, ..., num_experts)
        full_probs = keras.ops.softmax(clean_logits, axis=-1)
        expanded_probs = keras.ops.expand_dims(
            full_probs, axis=-2
        ) # (batch, ..., 1, num_experts)

        if training:
            raw_noise = self.w_noise(router_x) # (batch, ..., num_experts)
            noise_std = keras.ops.softplus(raw_noise)
            noise = noise_std * keras.random.normal(
                shape=keras.ops.shape(clean_logits)
            )
            logits = clean_logits + noise # (batch, ..., num_experts)
        else:
            logits = clean_logits

        # --- Noisy Top-k Gating Computation ---
        top_k_logits, top_k_indices = keras.ops.top_k(logits, k=self.k)
        # top_k_logits shape: (batch, ..., k)
        # top_k_indices shape: (batch, ..., k)

        # Expand indices and probabilities to scatter back to full expert dimension
        one_hot_mask = keras.ops.one_hot(
            top_k_indices, num_classes=self.num_experts
        )  # (batch, ..., k, num_experts)

        # Dense routing weights matrix: shape (batch, ..., num_experts)
        routing_weights = keras.ops.sum(
            expanded_probs * one_hot_mask, axis=-2
        )

        # For k > 1, renormalize probabilities across the selected top-k experts.
        # For k = 1, keep full softmax probability directly (Switch Transformer style)
        # so that the gate retains a non-zero task gradient.
        if self.k > 1:
            routing_weights = routing_weights / (
                keras.ops.sum(routing_weights, axis=-1, keepdims=True) + 1e-6
            )

        step_loss = keras.ops.zeros((), dtype=keras.ops.dtype(logits))

        # --- Losses (Calculated during training) ---
        if training:
            reduce_axes = tuple(range(keras.ops.ndim(full_probs) - 1))

            # 1. Auxiliary Load-Balancing Loss
            # P_e: mean full probability assigned to expert e prior 
            # to top-k cutoff
            p_e = keras.ops.mean(full_probs, axis=reduce_axes)

            # f_e: fraction of tokens routed to expert e 
            # (stop gradient to treat as fixed target)
            expert_mask = keras.ops.cast(
                keras.ops.greater(routing_weights, 0.0), dtype="float32"
            )
            f_e = keras.ops.stop_gradient(
                keras.ops.mean(expert_mask, axis=reduce_axes)
            )

            aux_loss = (
                self.aux_loss_weight
                * self.num_experts
                * keras.ops.sum(f_e * p_e)
            )

            # 2. Router Z-Loss (stabilizes logit magnitudes in mixed precision, 
            # computed on clean logits)
            log_z = keras.ops.logsumexp(clean_logits, axis=-1)
            z_loss = self.z_loss_weight * keras.ops.mean(
                keras.ops.square(log_z)
            )

            step_loss = aux_loss + z_loss
            # step loss shape: () scalar tensor

        # routing_weights shape: (batch, ..., num_experts)
        return routing_weights, step_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_experts": self.num_experts,
                "k": self.k,
                "aux_loss_weight": self.aux_loss_weight,
                "z_loss_weight": self.z_loss_weight,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)    