import keras

from experiments.kerass.layers.sequences.cells import HIDDEN_STATE_KEY
from experiments.kerass.layers.sequences.constants import STATE_KEY
from experiments.kerass.layers.gnn import instantiate_gnn_layer
from .GAPLSTMCell import GAPLSTMCell
from .AutoregressiveCell import AutoregressiveCell

class GAPLSTMWithMemoryCell(keras.layers.Layer, AutoregressiveCell):

    def __init__(
        self,
        states_dim: int,
        output_dim: int = None,

        gcnlstm_cell_config: dict = {},

        i_gnn_config: dict = {},
        g_gnn_config: dict = {},
        o_gnn_config: dict = {},
        im_gnn_config: dict = {},
        gm_gnn_config: dict = {},
        om_gnn_config: dict = {},

        return_state: bool = False,
        return_hidden_state: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.states_dim = states_dim
        self.output_dim = output_dim

        self.return_state = return_state
        self.return_hidden_state = return_hidden_state

        self.cell = GAPLSTMCell(
            states_dim=states_dim,
            output_dim=None,
            return_hidden_state=False,
            **gcnlstm_cell_config
        )

        self.i_gnn = instantiate_gnn_layer(
            i_gnn_config, channels=self.states_dim, name="i_gnn"
        )
        self.g_gnn = instantiate_gnn_layer(
            g_gnn_config, channels=self.states_dim, name="g_gnn"
        )
        self.o_gnn = instantiate_gnn_layer(
            o_gnn_config, channels=self.states_dim, name="o_gnn"
        )
        self.im_gnn = instantiate_gnn_layer(
            im_gnn_config, channels=self.states_dim, name="im_gnn"
        )
        self.gm_gnn = instantiate_gnn_layer(
            gm_gnn_config, channels=self.states_dim, name="gm_gnn"
        )
        self.om_gnn = instantiate_gnn_layer(
            om_gnn_config, channels=self.states_dim, name="om_gnn"
        )

        if self.output_dim is not None:
            self.output_dense = keras.layers.Dense(self.output_dim)

    def build(self, input_shape: tuple[keras.KerasTensor, keras.KerasTensor]):
        self.cell.build(input_shape)
        self.input_feature_dim = self.cell.input_feature_dim

        features_shape = input_shape[0]
        adj_shape = input_shape[1]
        states_shape = (features_shape[0], self.cell.n_nodes, self.cell.states_dim)
        new_shape = (states_shape, adj_shape)

        self.i_gnn.build(new_shape)
        self.g_gnn.build(new_shape)
        self.o_gnn.build(new_shape)
        self.im_gnn.build(new_shape)
        self.gm_gnn.build(new_shape)
        self.om_gnn.build(new_shape)

        if self.output_dim is not None:
            self.output_dense.build(states_shape)

        super().build(input_shape)

    @property
    def state_size(self) -> tuple[tuple[int]]:
        state_shape = (self.cell.n_nodes, self.cell.states_dim)
        return (
            state_shape,  # hidden_state
            state_shape,  # carry_state
            state_shape,  # memory_state
        )

    def get_initial_state(
        self, batch_size=None, dtype=None
    ) -> tuple[keras.KerasTensor]:
        dtype = dtype or keras.backend.floatx()
        h = keras.ops.zeros(
            (batch_size, self.cell.n_nodes, self.cell.states_dim), dtype=dtype
        )
        c = keras.ops.zeros(
            (batch_size, self.cell.n_nodes, self.cell.states_dim), dtype=dtype
        )
        m = keras.ops.zeros(
            (batch_size, self.cell.n_nodes, self.cell.states_dim), dtype=dtype
        )
        return h, c, m

    @property
    def log_size(self) -> dict[str, int]:
        log_size = {}
        if self.return_hidden_state:
            log_size[HIDDEN_STATE_KEY] = (
                self.cell.n_nodes, self.cell.states_dim
            )
        if self.return_state:
            log_size[STATE_KEY] = self.state_size

        return log_size

    @property
    def output_size(self) -> tuple[int]:
        if self.output_dim is not None:
            return (self.cell.n_nodes, self.output_dim)
        else:
            return (self.cell.n_nodes, self.cell.states_dim)

    def call(
        self, 
        inputs: tuple[keras.KerasTensor, keras.KerasTensor],
        states: tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor],
        training=None,
    ) -> tuple[
        keras.KerasTensor,
        tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor],
        dict[str, keras.KerasTensor],
        keras.KerasTensor,
    ]:
        _, adj = inputs

        _, next_state, _, step_loss = self.cell(
            inputs=inputs,
            states=states[:2],  # Pass only hidden_state and carry_state
            training=training,
        )

        h, c = next_state
        memory_state = states[-1]

        SA_ih, loss_ih = self.i_gnn((h, adj), training=training)
        SA_im, loss_im = self.im_gnn((memory_state, adj), training=training)
        SA_gh, loss_gh = self.g_gnn((h, adj), training=training)
        SA_gm, loss_gm = self.gm_gnn((memory_state, adj), training=training)
        SA_oh, loss_oh = self.o_gnn((h, adj), training=training)
        SA_om, loss_om = self.om_gnn((memory_state, adj), training=training)

        i_m = keras.ops.sigmoid(SA_ih + SA_im)
        g_m = keras.ops.sigmoid(SA_gh + SA_gm)
        o_m = keras.ops.sigmoid(SA_oh + SA_om)

        m = i_m * memory_state + (1.0 - i_m) * g_m
        h = m * o_m

        mem_gnn_loss = (
            loss_ih + loss_im + loss_gh + loss_gm + loss_oh + loss_om
        ) / 6.0
        step_loss = step_loss + mem_gnn_loss

        output = h
        log = {}

        if self.output_dim is not None:
            output = self.output_dense(h)

        if self.return_hidden_state:
            log[HIDDEN_STATE_KEY] = h

        if self.return_state:
            log[STATE_KEY] = (h, c, m)

        return output, (h, c, m), log, step_loss
