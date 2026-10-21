import keras

from experiments.kerass.layers.sequences.cells.constants import HIDDEN_STATE_KEY
from experiments.kerass.layers.sequences.constants import STATE_KEY
from .AutoregressiveCell import AutoregressiveCell

class GRUCell(keras.layers.Layer, AutoregressiveCell):

    def __init__(
        self,
        states_dim: int,
        n_nodes: int,
        output_dim: int = None,

        return_state: bool = False,
        return_hidden_state: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.n_nodes = n_nodes
        self.per_node_states_dim = states_dim
        self.return_state = return_state
        self.return_hidden_state = return_hidden_state

        states_dim = self.n_nodes * states_dim
        self.gru_cell = keras.layers.GRUCell(
            units=states_dim,
        )

        self.output_dim = output_dim
        if output_dim is not None:
            self.output_projection = keras.layers.Dense(
                units=self.output_dim,
                name="output_projection",
            )

    def build(self, input_shape):
        x = input_shape[0]

        B, N, F = x
        if N != self.n_nodes:
            raise ValueError(
                f"Expected input with {self.n_nodes} nodes, "
                f"but got {N} nodes."
            )

        self.gru_cell.build((B, N * F))
        if self.output_dim is not None:
            self.output_projection.build((B, N, self.per_node_states_dim))
        
        super().build(input_shape)

    @property
    def state_size(self) -> tuple[tuple[int], tuple[int]]:
        state_shape = (self.n_nodes, self.per_node_states_dim)
        return (state_shape,)

    def get_initial_state(
        self, batch_size=None, dtype=None
    ) -> tuple[keras.KerasTensor]:
        dtype = dtype or keras.backend.floatx()
        hidden_state = keras.ops.zeros(
            (batch_size, self.n_nodes, self.per_node_states_dim), 
            dtype=dtype
        )
        return (hidden_state,)

    @property
    def log_size(self) -> dict[str, int]:
        log_size = {}
        if self.return_hidden_state:
            log_size[HIDDEN_STATE_KEY] = (
                self.n_nodes, self.per_node_states_dim
            )
        if self.return_state:
            log_size[STATE_KEY] = self.state_size

        return log_size

    @property
    def output_size(self) -> tuple[int]:
        if self.output_dim is not None:
            return (self.n_nodes, self.output_dim,)
        else:
            return (self.n_nodes, self.per_node_states_dim,)

    def call(self, inputs, states, training=None):
        x, _ = inputs  # Unpack inputs and ignore adjacency matrix

        _, N, F = x.shape
        x = keras.ops.reshape(x, (-1, N * F))

        states = keras.tree.map_structure(
            lambda s: keras.ops.reshape(
                s, (-1, N * self.per_node_states_dim)
            ), states
        )

        output, new_states = self.gru_cell(x, states, training=training)

        new_states = keras.tree.map_structure(
            lambda s: keras.ops.reshape(
                s, (-1, N, self.per_node_states_dim)
            ), new_states
        )

        output = keras.ops.reshape(
            output, (-1, N, self.per_node_states_dim)
        )
        
        if self.output_dim is not None:
            output = self.output_projection(output)
        
        logs = {}
        if self.return_hidden_state:
            logs[HIDDEN_STATE_KEY] = new_states[0]
        if self.return_state:
            logs[STATE_KEY] = tuple(new_states)

        step_loss = keras.ops.zeros((), dtype=keras.ops.dtype(output))
        return output, new_states, logs, step_loss