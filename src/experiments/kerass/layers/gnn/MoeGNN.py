import keras
from typing import List, Dict, Tuple, Optional, Any

from .GCNSKIPCONN import GCNSKIPCONN
from .GraphConv import GraphConv
from .MoeGraphConv import MoeGraphConv
from .WeightedGraphConv import WeightedGraphConv
from .GAT import GAT

GNN_LAYERS = {
    "GCNSKIPCONN": GCNSKIPCONN,
    "spektral": GraphConv,
    "moe_spektral": MoeGraphConv,
    "weighted": WeightedGraphConv,
    "gat": GAT,
}

class MoeGNN(keras.layers.Layer):

    def __init__(
        self,
        channels: int,
        experts_configs: List[Dict[str, Any]],
        moe_router: keras.layers.Layer,
        pooling_layer: keras.layers.Layer,
        router_feature_indices: Optional[List[int]] = None,
        return_routing_weights: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not experts_configs:
            raise ValueError("The 'experts_configs' list cannot be empty.")

        self.channels = channels
        self.router_feature_indices = router_feature_indices
        self.return_routing_weights = return_routing_weights
        self.num_experts = len(experts_configs)
        self.moe_router = moe_router
        self.pooling_layer = pooling_layer
        
        self.experts = []
        for i, config in enumerate(experts_configs):
            config_copy = config.copy()  # Prevent in-place mutation bug
            gnn_type = config_copy.pop("gnn_type", None)
            
            gnn_layer_class = GNN_LAYERS.get(gnn_type, None)
            if gnn_layer_class is None:
                raise ValueError(
                    f"Invalid gnn_type '{gnn_type}'. "
                    f"Supported types are: {list(GNN_LAYERS.keys())}"
                )
            
            self.experts.append(
                gnn_layer_class(
                    **config_copy, 
                    channels=channels, 
                    name=f"gnn_expert_{i}"
                )
            )

    def build(self, input_shape):
        self.pooling_layer.build(input_shape)
        router_shape = self.pooling_layer.compute_output_shape(input_shape)

        if not self.moe_router.built:
            self.moe_router.build(router_shape)

        for expert in self.experts:
            if not expert.built:
                expert.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
        training: bool = False,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        x, adj = inputs

        x_shape = keras.ops.shape(x)
        dtype = keras.ops.dtype(x)

        def bypass_expert():
            zero_out = keras.ops.zeros(
                x_shape[:-1] + (self.channels,), dtype=dtype
            )
            return zero_out, keras.ops.zeros((), dtype=dtype)

        if self.router_feature_indices is not None:
            x = keras.ops.take(x, indices=self.router_feature_indices, axis=-1)

        pooled_x = self.pooling_layer((x, adj), training=training)
        routing_weights, step_loss = self.moe_router(
            pooled_x, training=training
        ) # routing_weights shape: (B, E)

        expert_activity = keras.ops.sum(
            routing_weights,
            axis=(0, )
        )  # Shape: (E,)

        total_loss = step_loss
        accum_output = keras.ops.zeros(
            x_shape[:-1] + (self.channels,), dtype=dtype
        )

        for idx, expert in enumerate(self.experts):
            def compute_expert():
                return expert((x, adj), training=training)

            is_active = keras.ops.greater(expert_activity[idx], 0.0)
            
            exp_out, expert_loss = keras.ops.cond(
                is_active, compute_expert, bypass_expert
            )

            loss_weight = expert_activity[idx] / (
                keras.ops.sum(expert_activity) + 1e-6
            )
            total_loss = total_loss + (expert_loss * loss_weight)

            w_e = routing_weights[:, idx]  # Shape: (B, )
            w_e = keras.ops.expand_dims(w_e, axis=-1)  # Shape: (B, 1)
            w_e = keras.ops.expand_dims(w_e, axis=-1)  # Shape: (B, 1, 1)

            accum_output = accum_output + (w_e * exp_out)

        return accum_output, total_loss