import keras

from experiments.kerass.layers.sequences.cells import HIDDEN_STATE_KEY
from experiments.kerass.layers.sequences.constants import STATE_KEY

class BaseLSTMCell(keras.layers.Layer):

    def __init__(
        self,
        states_dim: int,
        output_dim: int = None,

        return_state: bool = False,
        return_hidden_state: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.states_dim = states_dim
        self.output_dim = output_dim
        self.return_state = return_state
        self.return_hidden_state = return_hidden_state
        
        self.lstm_cell = keras.layers.LSTMCell(
            units=self.states_dim,
        )
        if output_dim is not None:
            self.output_projection = keras.layers.Dense(
                units=output_dim,
                name="output_projection",
            )

    def build(self, input_shape):
        B = input_shape[0]

        self.lstm_cell.build(input_shape)

        if self.output_dim is not None:
            self.output_projection.build((B, self.states_dim))

        super().build(input_shape)

    @property
    def state_size(self) -> tuple[tuple[int], tuple[int]]:
        state_shape = (self.states_dim,)
        return (
            state_shape,  # hidden_state
            state_shape,  # carry_state
        )

    def get_initial_state(
        self, batch_size=None, dtype=None
    ) -> tuple[keras.KerasTensor, keras.KerasTensor]:
        dtype = dtype or keras.backend.floatx()
        h = keras.ops.zeros((batch_size, self.states_dim), dtype=dtype)
        c = keras.ops.zeros((batch_size, self.states_dim), dtype=dtype)
        return h, c

    @property
    def log_size(self) -> dict[str, int]:
        log_size = {}
        if self.return_hidden_state:
            log_size[HIDDEN_STATE_KEY] = (self.states_dim,)
        if self.return_state:
            log_size[STATE_KEY] = self.state_size

        return log_size

    @property
    def output_size(self) -> tuple[int]:
        if self.output_dim is not None:
            return (self.output_dim,)
        else:
            return (self.states_dim,)

    def call(self, inputs, states, training=None):
        output, new_states = self.lstm_cell(inputs, states, training=training)

        logs = {}
        if self.return_hidden_state:
            logs[HIDDEN_STATE_KEY] = new_states[0]
        if self.return_state:
            logs[STATE_KEY] = tuple(new_states)

        if self.output_projection is not None:
            output = self.output_projection(output)

        step_loss = keras.ops.zeros((), dtype=keras.ops.dtype(output))
        return output, new_states, logs, step_loss
