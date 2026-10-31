import keras
from typing import Tuple


class ShazeerMoeRouter(keras.layers.Layer):
    """Noisy Top-k MoE Router based on Shazeer et al. (2017).

    Guarantees non-zero gradients to w_noise by evaluating softmax over the
    selected noisy logits. Renormalization happens organically over top-k.
    """

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
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        dtype = keras.ops.dtype(inputs)
        clean_logits = self.w_gate(inputs)  # (batch, ..., num_experts)

        # 1. Noisy Logits Computation
        if training:
            raw_noise = self.w_noise(inputs)
            noise_std = keras.ops.softplus(raw_noise)
            noise = noise_std * keras.random.normal(
                shape=keras.ops.shape(clean_logits), dtype=dtype
            )
            logits = clean_logits + noise
        else:
            logits = clean_logits

        # 2. Top-k Selection on Noisy Logits
        # top_k_logits retains full gradient flow to BOTH w_gate and w_noise!
        top_k_logits, top_k_indices = keras.ops.top_k(logits, k=self.k)

        # 3. Probabilities evaluated on chosen top-k noisy logits
        # Naturally normalizes to 1.0 across selected experts (k=1 yields 1.0)
        top_k_gates = keras.ops.softmax(top_k_logits, axis=-1)  # (batch, ..., k)

        # 4. Memory-Efficient Projection:
        # Instead of expanding full probabilities, scale the one-hot indices directly
        one_hot_mask = keras.ops.one_hot(
            top_k_indices, num_classes=self.num_experts, dtype=dtype
        )  # (batch, ..., k, num_experts)

        routing_weights = keras.ops.sum(
            keras.ops.expand_dims(top_k_gates, axis=-1) * one_hot_mask,
            axis=-2,
        )  # (batch, ..., num_experts)

        step_loss = keras.ops.zeros((), dtype=dtype)

        # 5. Loss Calculation (Auxiliary + Z-Loss)
        if training:
            full_clean_probs = keras.ops.softmax(clean_logits, axis=-1)
            reduce_axes = tuple(range(keras.ops.ndim(full_clean_probs) - 1))

            # P_e: average clean probability directed to expert e
            p_e = keras.ops.mean(full_clean_probs, axis=reduce_axes)

            # f_e: fraction of tokens actively dispatched to expert e
            expert_mask = keras.ops.cast(
                keras.ops.greater(routing_weights, 0.0), dtype=dtype
            )
            f_e = keras.ops.stop_gradient(
                keras.ops.mean(expert_mask, axis=reduce_axes)
            )

            # Load balancing loss
            aux_loss = (
                self.aux_loss_weight
                * self.num_experts
                * keras.ops.sum(f_e * p_e)
            )

            # Router Z-loss (computed on clean logits to stabilize mixed-precision)
            log_z = keras.ops.logsumexp(clean_logits, axis=-1)
            z_loss = self.z_loss_weight * keras.ops.mean(
                keras.ops.square(log_z)
            )

            step_loss = aux_loss + z_loss

        return routing_weights, step_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_experts": self.num_experts,
                "top_k": self.k,
                "aux_loss_weight": self.aux_loss_weight,
                "z_loss_weight": self.z_loss_weight,
            }
        )
        return config