from collections import defaultdict

import keras

from experiments.kerass.layers.sequences.constants import STATE_KEY

from .AutoregressiveCell import AutoregressiveCell

class StackedCells(keras.layers.Layer, AutoregressiveCell):
    def __init__(
        self, 
        cells: list[keras.layers.Layer], 
        return_state: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not cells:
            raise ValueError("The 'cells' list cannot be empty.")
        self.cells = cells
        self.return_state = return_state

    def build(self, input_shape):
        features_shape, adjacency_shape = input_shape
        current_features_shape = features_shape

        self.input_feature_dim = features_shape[-1]
        self.n_nodes = features_shape[-2]

        for cell in self.cells:
            cell_input_shape = (current_features_shape, adjacency_shape)
            if not cell.built:
                cell.build(cell_input_shape)
            current_features_shape = (
                features_shape[0], *cell.output_size
            )

        super().build(input_shape)

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self.cells)

    @property
    def output_size(self):
        return self.cells[-1].output_size

    @property
    def log_size(self):
        log_size = defaultdict(list)
        for cell in self.cells:
            for key, value in cell.log_size.items():
                log_size[key].append(value)

        if self.return_state:
            log_size[STATE_KEY] = tuple(cell.state_size for cell in self.cells)

        return log_size

    def get_initial_state(self, batch_size=None, dtype=None):
        return tuple(
            cell.get_initial_state(batch_size=batch_size, dtype=dtype)
            for cell in self.cells
        )

    def call(
        self, 
        inputs: tuple[keras.KerasTensor, keras.KerasTensor],
        states: tuple[tuple[keras.KerasTensor]], 
        training = None,
    ):

        x, adj = inputs
        new_states = []
        stacked_logs = defaultdict(list)
        step_loss = keras.ops.zeros((), dtype=keras.backend.floatx())

        for index, cell in enumerate(self.cells):
            cell_states = states[index]
            x, new_cell_states, logs, cell_step_loss = cell(
                (x, adj), cell_states, training=training
            )
            new_states.append(new_cell_states)
            for key, value in logs.items():
                stacked_logs[key].append(value)

        if self.return_state:
            stacked_logs[STATE_KEY] = tuple(new_states)

        return x, tuple(new_states), stacked_logs, step_loss
