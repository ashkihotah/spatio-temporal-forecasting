import keras

class HMoeAdapter(keras.layers.Layer):

    def __init__(
        self,
        cell: keras.layers.Layer,
        moe_router: keras.layers.Layer,
        return_routing_weights: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cell = cell
        self.moe_router = moe_router
        self.n_head_experts = moe_router.num_experts
        self.return_routing_weights = return_routing_weights

        if self.n_head_experts < 2:
            raise ValueError("n_head_experts must be at least 2")

        self.output_denses = [
            keras.layers.Dense(self.cell.output_dim) 
            for _ in range(self.n_head_experts)
        ]

    def build(self, input_shape):
        if not self.cell.built:
            self.cell.build(input_shape)

        # Assuming the first element is the hidden state
        hidden_state_shape = self.cell.state_size[0]

        if not self.moe_router.built:
            self.moe_router.build(hidden_state_shape)
        
        for i in range(self.n_head_experts):
            self.output_denses[i].build(hidden_state_shape)

        super().build(input_shape)

    @property
    def state_size(self):
        return self.cell.state_size
    
    def get_initial_state(
        self, batch_size=None, dtype=None
    ):
        return self.cell.get_initial_state(batch_size=batch_size, dtype=dtype)

    @property
    def log_size(self) -> dict[str, int]:
        log_size = self.cell.log_size
        if self.return_routing_weights:
            log_size["routing_weights"] = (self.cell.n_nodes, self.moe_router.num_experts)
        return log_size

    @property
    def output_size(self) -> tuple[int]:
        return self.cell.output_size

    def get_cell_input_shape(self, prev_outputs_shape, current_inputs_shape):
        return self.cell.get_cell_input_shape(prev_outputs_shape, current_inputs_shape)
    
    def get_cell_input(self, prev_outputs, current_inputs):
        return self.cell.get_cell_input(prev_outputs, current_inputs)
    
    def call(self, inputs, states, training=None):
        output, new_states, log, step_loss = self.cell(
            inputs, states, training=training
        )
        h = new_states[0]  # Assuming the first element is the hidden state

        if self.cell.output_dim is not None:
            routing_weights, head_loss = self.moe_router(h, training=training)
            step_loss = step_loss + head_loss

            expert_outputs = keras.ops.stack([
                expert(h) for expert in self.output_denses
            ], axis=-2)  # (batch, n_nodes, n_head_experts, output_dim)
            output = keras.ops.sum(
                keras.ops.expand_dims(routing_weights, -1) * expert_outputs,
                axis=-2
            )

            if self.return_routing_weights:
                log["routing_weights"] = routing_weights

        return output, new_states, log, step_loss

    def clone_with_new_cell(self, new_cell: keras.layers.Layer, **kwargs):
        return HMoeAdapter(
            cell=new_cell,
            moe_router=self.moe_router,
            return_routing_weights=self.return_routing_weights,
            **kwargs
        )