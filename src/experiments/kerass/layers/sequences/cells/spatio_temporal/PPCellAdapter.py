import keras

class PPCellAdapter(keras.layers.Layer):

    def __init__(
        self,
        cell: keras.layers.Layer,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cell = cell

    def build(self, input_shape):
        if not self.cell.built:
            self.cell.build(input_shape)
        super().build(input_shape)

    @property
    def state_size(self):
        return self.cell.state_size

    def get_initial_state(self, batch_size=None, dtype=None):
        return self.cell.get_initial_state(batch_size, dtype)

    @property
    def log_size(self):
        return self.cell.log_size

    @property
    def output_size(self):
        return self.cell.output_size

    def get_cell_input_shape(self, prev_outputs_shape, current_inputs_shape):
        return self.cell.get_cell_input_shape(prev_outputs_shape, current_inputs_shape)

    def get_cell_input(self, prev_outputs, current_inputs):
        return self.cell.get_cell_input(prev_outputs, current_inputs)

    def call(self, inputs, states, training=None):
        raise NotImplementedError(
            "The `call` method must be implemented in " \
            "subclasses of CellAdapter."
        )

    def clone_with_new_cell(self, new_cell: keras.layers.Layer, **kwargs):
        raise NotImplementedError(
            "The `clone_with_new_cell` method must be implemented in " \
            "subclasses of CellAdapter."
        )