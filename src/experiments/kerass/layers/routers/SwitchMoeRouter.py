import keras
from typing import Tuple


class SwitchMoeRouter(keras.layers.Layer):
    """Switch Transformer Style Router based on Fedus et al. (2021).

    Eliminates w_noise. Exploration is done via input-independent jitter.
    Preserves raw clean probabilities for k=1 to ensure smooth task gradient flow.
    """

    def __init__(
        self,
        n_experts: int,
        top_k: int = 1,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        jitter_eps: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = n_experts
        self.k = min(top_k, self.num_experts)
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight
        self.jitter_eps = jitter_eps

        # Single projection gate: no noise network needed
        self.w_gate = keras.layers.Dense(
            self.num_experts, use_bias=False, name="router_gate"
        )

    def build(self, input_shape):
        self.w_gate.build(input_shape)
        super().build(input_shape)

    def call(
        self, inputs: keras.KerasTensor, training: bool = False
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        dtype = keras.ops.dtype(inputs)
        clean_logits = self.w_gate(inputs)  # (batch, ..., num_experts)

        # 1. Switch Transformer Jittering
        # Exploration without large variance logit distortions
        if training and self.jitter_eps > 0.0:
            jitter = keras.random.uniform(
                shape=keras.ops.shape(clean_logits),
                minval=1.0 - self.jitter_eps,
                maxval=1.0 + self.jitter_eps,
                dtype=dtype,
            )
            logits = clean_logits * jitter
        else:
            logits = clean_logits

        # 2. Discrete Expert Selection
        _, top_k_indices = keras.ops.top_k(logits, k=self.k)

        # 3. Softmax on Clean Logits
        full_clean_probs = keras.ops.softmax(clean_logits, axis=-1)

        # Gather clean probabilities for chosen indices
        top_k_clean_probs = keras.ops.take_along_axis(
            full_clean_probs, top_k_indices, axis=-1
        )  # (batch, ..., k)

        # Normalization rule:
        # k > 1: Renormalize over the top-k subset
        # k = 1: Keep full softmax probability (Switch Transformer gradient design)
        if self.k > 1:
            top_k_gates = top_k_clean_probs / (
                keras.ops.sum(top_k_clean_probs, axis=-1, keepdims=True) + 1e-6
            )
        else:
            top_k_gates = top_k_clean_probs

        # 4. Efficient Scatter to Dense Matrix
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
            reduce_axes = tuple(range(keras.ops.ndim(full_clean_probs) - 1))

            p_e = keras.ops.mean(full_clean_probs, axis=reduce_axes)

            expert_mask = keras.ops.cast(
                keras.ops.greater(routing_weights, 0.0), dtype=dtype
            )
            f_e = keras.ops.stop_gradient(
                keras.ops.mean(expert_mask, axis=reduce_axes)
            )

            aux_loss = (
                self.aux_loss_weight
                * self.num_experts
                * keras.ops.sum(f_e * p_e)
            )

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
                "jitter_eps": self.jitter_eps,
            }
        )
        return config