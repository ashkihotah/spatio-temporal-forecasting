import keras

from .CellAdapter import CellAdapter

class HMoeAdapter(CellAdapter):

    def __init__(
        self,
        moe_router: keras.layers.Layer,
        return_routing_weights: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.moe_router = moe_router
        self.n_head_experts = moe_router.num_experts
        self.return_routing_weights = return_routing_weights

        if self.n_head_experts < 2:
            raise ValueError("n_head_experts must be at least 2")

    def build(self, input_shape):
        if not self.cell.built:
            self.cell.build(input_shape)

        hidden_dim = self.cell.state_size[0][-1]
        output_dim = self.cell.output_dim

        if not self.moe_router.built:
            # Build router with shape: (..., hidden_dim)
            self.moe_router.build((None, hidden_dim))

        # Single consolidated 3D parameter tensor for all E experts
        # Eliminates E separate Dense layers and kernel launch overhead
        self.expert_kernel = self.add_weight(
            name="expert_kernels",
            shape=(self.n_head_experts, hidden_dim, output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.expert_bias = self.add_weight(
            name="expert_biases",
            shape=(self.n_head_experts, output_dim),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    @property
    def log_size(self) -> dict:
        log_size = getattr(self.cell, "log_size", {}).copy()
        if self.return_routing_weights:
            log_size["routing_weights"] = (self.cell.n_nodes, self.n_head_experts)
        return log_size

    def call(self, inputs, states, training=None):
        output, new_states, log, step_loss = self.cell(
            inputs, states, training=training
        )
        h = new_states[0]  # Shape: (batch, n_nodes, hidden_dim)

        if self.cell.output_dim is not None:
            # routing_weights: (batch, n_nodes, n_head_experts)
            routing_weights, head_loss = self.moe_router(h, training=training)
            step_loss = step_loss + head_loss

            # active_weights: (batch, n_nodes, k)
            # active_indices: (batch, n_nodes, k)
            active_weights, active_indices = keras.ops.top_k(
                routing_weights,
                k=self.moe_router.k,
            )

            # chosen_kernels shape: (batch, n_nodes, k, hidden_dim, output_dim)
            # chosen_biases shape:  (batch, n_nodes, k, output_dim)
            chosen_kernels = keras.ops.take(
                self.expert_kernel, active_indices, axis=0
            )
            chosen_biases = keras.ops.take(
                self.expert_bias, active_indices, axis=0
            )

            # O(k) Fused Linear Projection
            # Compute h * W_k + b_k exclusively for selected experts
            # h expanded: (batch, n_nodes, 1, hidden_dim)
            h_expanded = keras.ops.expand_dims(h, axis=-2)
            
            # expert_outputs shape: (batch, n_nodes, k, output_dim)
            expert_outputs = (
                keras.ops.matmul(
                    # (..., 1, 1, hidden)
                    keras.ops.expand_dims(h_expanded, axis=-2),
                    # (..., k, hidden, out)
                    chosen_kernels
                )
            )
            # Squeeze matrix multiplication dimension
            expert_outputs = keras.ops.squeeze(expert_outputs, axis=-2) 
            expert_outputs += chosen_biases

            # Weighted Mixture Reduction over Top-k
            # active_weights expanded: (batch, n_nodes, k, 1)
            output = keras.ops.sum(
                keras.ops.expand_dims(active_weights, axis=-1) * expert_outputs,
                axis=-2
            )  # Output shape: (batch, n_nodes, output_dim)

            if self.return_routing_weights:
                log["routing_weights"] = routing_weights

        return output, new_states, log, step_loss

    def clone_with_new_cell(self, new_cell: keras.layers.Layer, **kwargs):
        return HMoeAdapter(
            cell=new_cell,
            moe_router=self.moe_router,
            return_routing_weights=self.return_routing_weights,
            **kwargs,
        )