import keras
from typing import List, Tuple, Dict, Any, Optional

from experiments.kerass.layers.sequences.cells.constants import HIDDEN_STATE_KEY

from .AutoregressiveCell import AutoregressiveCell

class MoeCell(keras.layers.Layer, AutoregressiveCell):

    def __init__(
        self,
        experts: List[keras.layers.Layer],
        pooling_layer: keras.layers.Layer,
        moe_router: keras.layers.Layer,
        router_feature_indices: Optional[List[int]] = None,
        return_routing_weights: bool = False,
        return_hidden_state: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not experts:
            raise ValueError("The 'experts' list cannot be empty.")

        self.experts = experts
        self.pooling_layer = pooling_layer
        self.num_experts = len(experts)
        self.moe_router = moe_router
        self.router_feature_indices = router_feature_indices
        self.return_routing_weights = return_routing_weights
        self.return_hidden_state = return_hidden_state

    def build(self, input_shape):
        adj_shape = input_shape[1]
        self.n_nodes = adj_shape[0]

        if not self.pooling_layer.built:
            self.pooling_layer.build(input_shape)

        router_shape = self.pooling_layer.compute_output_shape(input_shape)
        if not self.moe_router.built:
            self.moe_router.build(router_shape)

        for expert in self.experts:
            if not expert.built:
                expert.build(input_shape)

        super().build(input_shape)

    @property
    def state_size(self):
        return self.experts[0].state_size

    @property
    def output_size(self):
        return self.experts[0].output_size

    @property
    def log_size(self):
        log_size = {}
        if self.return_routing_weights:
            log_size["routing_weights"] = (self.num_experts,)

        for idx, expert in enumerate(self.experts):
            if hasattr(expert, "log_size"):
                for log_name, log_shape in expert.log_size.items():
                    log_size[f"expert_{idx}_{log_name}"] = log_shape

        if self.return_hidden_state:
            log_size["hidden_state"] = self.state_size

        return log_size

    def get_initial_state(self, batch_size=None, dtype=None):
        return self.experts[0].get_initial_state(
            batch_size=batch_size, dtype=dtype
        )
    
    def call(
        self,
        inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
        states: Any,
        training: bool = False,
    ) -> Tuple[Any, Any, Dict[str, keras.KerasTensor], keras.KerasTensor]:
        x, adj = inputs

        dtype = keras.ops.dtype(x)
        batch_size = keras.ops.shape(x)[0]

        if self.router_feature_indices is not None:
            x = keras.ops.take(x, indices=self.router_feature_indices, axis=-1)
        
        pooled_x = self.pooling_layer((x, adj), training=training)
        routing_weights, router_loss = self.moe_router(
            pooled_x, training=training
        )
        # routing_weights: (B, E)

        # Align routing dimensions for broadcasting: Expand to (B, 1, E, 1)
        routing_weights = keras.ops.expand_dims(
            keras.ops.expand_dims(routing_weights, axis=1), axis=-1
        )

        expert_activity = keras.ops.sum(
            routing_weights, 
            axis=(0, 1, 3)
        ) # (E,)

        accum_out = keras.tree.map_shape_structure(
            lambda shape: keras.ops.zeros(
                shape, dtype=dtype
            ),
            self.output_size,
        )
        accum_state = keras.tree.map_shape_structure(
            lambda shape: keras.ops.zeros(
                shape, dtype=dtype
            ),
            self.state_size,
        )

        combined_log = {}
        total_loss = router_loss

        for idx, expert in enumerate(self.experts):
            is_active = keras.ops.greater(expert_activity[idx], 0.0)

            def compute_expert():
                return expert((x, adj), states, training=training)

            def bypass_expert():
                expert_output_size = expert.output_size
                expert_state_size = expert.state_size
                expert_log_size = expert.log_size
        
                zero_out = keras.tree.map_shape_structure(
                    lambda shape: keras.ops.zeros(
                        (batch_size,) + shape, dtype=dtype
                    ),
                    expert_output_size
                )
                zero_state = keras.tree.map_shape_structure(
                    lambda shape: keras.ops.zeros(
                        (batch_size,) + shape, dtype=dtype
                    ),
                    expert_state_size
                )
                zero_log = keras.tree.map_shape_structure(
                    lambda shape: keras.ops.zeros(
                        (batch_size,) + shape, dtype=dtype
                    ),
                    expert_log_size
                )
        
                zero_loss = keras.ops.zeros((), dtype=dtype)
    
                return zero_out, zero_state, zero_log, zero_loss
            
            exp_out, exp_state, exp_log, expert_loss = keras.ops.cond(
                is_active, compute_expert, bypass_expert
            )

            # Accumulate Weighted Loss
            loss_weight = expert_activity[idx] / (
                keras.ops.sum(expert_activity) + 1e-6
            )
            total_loss = total_loss + (expert_loss * loss_weight)

            w_e = routing_weights[..., idx, :]  # Shape: (batch, 1, 1)

            weighted_state = keras.tree.map_structure(
                lambda state: state * w_e, exp_state
            )

            accum_out = keras.tree.map_structure(
                lambda out, val: out + val,
                accum_out, exp_out * w_e,
            )
            accum_state = keras.tree.map_structure(
                lambda state, val: state + val,
                accum_state, weighted_state,
            )

            for log_k, log_v in exp_log.items():
                combined_log[f"expert_{idx}_{log_k}"] = log_v

        if self.return_hidden_state:
            combined_log[HIDDEN_STATE_KEY] = accum_state

        if self.return_routing_weights:
            combined_log["routing_weights"] = keras.ops.squeeze(
                routing_weights, axis=(1, 3) 
                # TODO: Check compatibility with routing weights plot
            )  # Shape: (B, E)

        return accum_out, accum_state, combined_log, total_loss