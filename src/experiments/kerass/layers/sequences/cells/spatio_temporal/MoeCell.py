import keras

from experiments.kerass.layers.routers.MoeRouter import MoeRouter
from .AutoregressiveCell import AutoregressiveCell

class MoeCell(keras.layers.Layer, AutoregressiveCell):
    """Keras 3 backend-agnostic Mixture of Experts (MoE) cell wrapper.
    """

    def __init__(
        self,
        experts: list[keras.layers.Layer],
        top_k: int = 1,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        router_feature_indices: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not experts:
            raise ValueError("The 'experts' list cannot be empty.")

        self.experts = experts
        self.router_feature_indices = router_feature_indices
        
        self.num_experts = len(experts)
        self.moe_router = MoeRouter(
            n_experts=self.num_experts,
            top_k=top_k,
            aux_loss_weight=aux_loss_weight,
            z_loss_weight=z_loss_weight,
        )

    def build(self, input_shape):
        features_shape, _ = input_shape
        self.input_feature_dim = features_shape[-1]
        self.n_nodes = features_shape[1]

        if self.router_feature_indices is None:
            moe_router_input_shape = (
                features_shape[0], self.n_nodes, 
                self.input_feature_dim
            )
        else:
            moe_router_input_shape = (
                features_shape[0], self.n_nodes, 
                len(self.router_feature_indices)
            )        

        # Build each expert cell
        for expert in self.experts:
            expert.build(input_shape)

        self.moe_router.build(moe_router_input_shape)

        super().build(input_shape)

    @property
    def state_size(self):
        return self.experts[0].state_size

    @property
    def output_size(self):
        return self.experts[0].output_size

    @property
    def log_size(self):
        log_size = {
            "routing_weights": (self.n_nodes, self.num_experts)
        }
        for idx, expert in enumerate(self.experts):
            for log_name, log_shape in expert.log_size.items():
                log_size[f"expert_{idx}_{log_name}"] = log_shape
                log_size[log_name] = log_shape
        return log_size

    def get_initial_state(self, batch_size=None, dtype=None):
        return self.experts[0].get_initial_state(
            batch_size=batch_size, dtype=dtype
        )

    def _select_router_features(self, x: keras.KerasTensor) -> keras.KerasTensor:
        if self.router_feature_indices is not None:
            return keras.ops.take(
                x, indices=self.router_feature_indices, axis=-1
            )
        return x

    def call(
        self,
        inputs: tuple[keras.KerasTensor, keras.KerasTensor],
        states: tuple[keras.KerasTensor, keras.KerasTensor],
        training: bool = False,
    ) -> tuple[
        keras.KerasTensor,
        tuple[keras.KerasTensor, keras.KerasTensor],
        dict[str, keras.KerasTensor],
        keras.KerasTensor,
    ]:
        x, adj = inputs

        # 1. Extract specified router features & calculate top-k routing weights
        x_router = self._select_router_features(x)
        # routing_weights, step_loss = self._compute_routing_and_losses(
        #     x_router, training=training
        # ) 
        routing_weights, step_loss = self.moe_router(
            x_router, training=training
        )

        # routing_weights shape: (batch, n_nodes, num_experts)

        accum_out = keras.tree.map_shape_structure(
            lambda shape: keras.ops.zeros(
                shape, dtype=x.dtype
            ),
            self.output_size,
        )
        accum_state = keras.tree.map_shape_structure(
            lambda shape: keras.ops.zeros(
                shape, dtype=x.dtype
            ),
            self.state_size,
        )
        combined_log = {}

        # 2. Forward pass through experts and blend active outputs
        for idx, expert in enumerate(self.experts):
            w_e = routing_weights[
                :, :, idx : idx + 1
            ]  # shape (batch, n_nodes, 1)

            # Execute expert
            exp_out, state, exp_log, expert_loss = expert(
                (x, adj), states, training=training
            )
            # exp_out shape: (batch, n_nodes, output_dim)
            step_loss = step_loss + expert_loss

            w_e__state = keras.tree.map_structure(
                lambda out: out * w_e, state
            )

            # Accumulate weighted outputs and state updates
            accum_out = keras.tree.map_structure(
                lambda out, val: out + val,
                accum_out, exp_out * w_e,
            )
            accum_state = keras.tree.map_structure(
                lambda state, val: state + val,
                accum_state, w_e__state,
            )

            # Collect cell logs
            for log_k, log_v in exp_log.items():
                combined_log[f"expert_{idx}_{log_k}"] = log_v
                weighted_log = log_v * w_e
                if log_k in combined_log:
                    combined_log[log_k] = combined_log[log_k] + weighted_log
                else:
                    combined_log[log_k] = weighted_log

        combined_log["routing_weights"] = routing_weights

        return accum_out, accum_state, combined_log, step_loss