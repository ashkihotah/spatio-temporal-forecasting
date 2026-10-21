import keras

from experiments.kerass.layers.sequences.cells import HIDDEN_STATE_KEY
from experiments.kerass.layers.sequences.constants import STATE_KEY
from experiments.kerass.layers.gnn import instantiate_gnn_layer
from .AutoregressiveCell import AutoregressiveCell

class GAPLSTMCell(keras.layers.Layer, AutoregressiveCell):

    def __init__(
        self,
        states_dim: int,
        output_dim: int = None,

        fx_gnn_config: dict = {},
        fh_gnn_config: dict = {},
        ix_gnn_config: dict = {},
        ih_gnn_config: dict = {},
        cx_gnn_config: dict = {},
        ch_gnn_config: dict = {},
        ox_gnn_config: dict = {},
        oh_gnn_config: dict = {},

        return_state: bool = False,
        return_hidden_state: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.states_dim = states_dim
        self.output_dim = output_dim

        self.return_state = return_state
        self.return_hidden_state = return_hidden_state

        self.fx_gnn = instantiate_gnn_layer(
            fx_gnn_config, channels=self.states_dim, name="fx_gnn"
        )
        self.fh_gnn = instantiate_gnn_layer(
            fh_gnn_config, channels=self.states_dim, name="fh_gnn"
        )
        self.ix_gnn = instantiate_gnn_layer(
            ix_gnn_config, channels=self.states_dim, name="ix_gnn"
        )
        self.ih_gnn = instantiate_gnn_layer(
            ih_gnn_config, channels=self.states_dim, name="ih_gnn"
        )
        self.cx_gnn = instantiate_gnn_layer(
            cx_gnn_config, channels=self.states_dim, name="cx_gnn"
        )
        self.ch_gnn = instantiate_gnn_layer(
            ch_gnn_config, channels=self.states_dim, name="ch_gnn"
        )
        self.ox_gnn = instantiate_gnn_layer(
            ox_gnn_config, channels=self.states_dim, name="ox_gnn"
        )
        self.oh_gnn = instantiate_gnn_layer(
            oh_gnn_config, channels=self.states_dim, name="oh_gnn"
        )

        if self.output_dim is not None:
            self.output_dense = keras.layers.Dense(self.output_dim)

    def build(self, input_shape: tuple[keras.KerasTensor, keras.KerasTensor]):
        features_shape = input_shape[0]
        adj_shape = input_shape[1]
        self.input_feature_dim = features_shape[-1]
        self.n_nodes = features_shape[1]

        hidden_state_shape = (features_shape[0], self.n_nodes, self.states_dim)

        self.fx_gnn.build(input_shape)
        self.fh_gnn.build((hidden_state_shape, adj_shape))
        self.ix_gnn.build(input_shape)
        self.ih_gnn.build((hidden_state_shape, adj_shape))
        self.cx_gnn.build(input_shape)
        self.ch_gnn.build((hidden_state_shape, adj_shape))
        self.ox_gnn.build(input_shape)
        self.oh_gnn.build((hidden_state_shape, adj_shape))

        if self.output_dim is not None:
            self.output_dense.build(hidden_state_shape)

        super().build(input_shape)

    @property
    def state_size(self) -> tuple[tuple[int], tuple[int]]:
        state_shape = (self.n_nodes, self.states_dim)
        return (
            state_shape,  # hidden_state
            state_shape,  # carry_state
        )

    def get_initial_state(
        self, batch_size=None, dtype=None
    ) -> tuple[keras.KerasTensor, keras.KerasTensor]:
        dtype = dtype or keras.backend.floatx()
        h = keras.ops.zeros(
            (batch_size, self.n_nodes, self.states_dim), dtype=dtype
        )
        c = keras.ops.zeros(
            (batch_size, self.n_nodes, self.states_dim), dtype=dtype
        )
        return h, c

    @property
    def log_size(self) -> dict[str, int]:
        log_size = {}
        if self.return_hidden_state:
            log_size[HIDDEN_STATE_KEY] = (self.n_nodes, self.states_dim)
        if self.return_state:
            log_size[STATE_KEY] = self.state_size

        return log_size

    @property
    def output_size(self) -> tuple[int]:
        output_shape = (self.n_nodes, self.states_dim)
        if self.output_dim is not None:
            output_shape = (self.n_nodes, self.output_dim)

        return output_shape

    def call(
        self, 
        inputs: tuple[keras.KerasTensor, keras.KerasTensor],
        states: tuple[keras.KerasTensor, keras.KerasTensor],
        training=None,
    ) -> tuple[
        keras.KerasTensor,
        tuple[keras.KerasTensor, keras.KerasTensor],
        dict[str, keras.KerasTensor],
        keras.KerasTensor,
    ]:
        x, adj = inputs

        hidden_state = states[0]
        carry_state = states[1]

        # Call all 8 GNN projections and extract their individual router losses
        fx, loss_fx = self.fx_gnn((x, adj), training=training)
        fh, loss_fh = self.fh_gnn((hidden_state, adj), training=training)
        ix, loss_ix = self.ix_gnn((x, adj), training=training)
        ih, loss_ih = self.ih_gnn((hidden_state, adj), training=training)
        cx, loss_cx = self.cx_gnn((x, adj), training=training)
        ch, loss_ch = self.ch_gnn((hidden_state, adj), training=training)
        ox, loss_ox = self.ox_gnn((x, adj), training=training)
        oh, loss_oh = self.oh_gnn((hidden_state, adj), training=training)

        f = keras.ops.sigmoid(fx + fh)
        i = keras.ops.sigmoid(ix + ih)
        o = keras.ops.sigmoid(ox + oh)
        c = f * carry_state + i * keras.ops.tanh(cx + ch)
        h = o * keras.ops.tanh(c)

        log = {}
        output = h

        # Normalize step losses across all 8 GNN gate layers
        step_loss = (
            loss_fx + loss_fh + loss_ix + loss_ih +
            loss_cx + loss_ch + loss_ox + loss_oh
        ) / 8.0

        if self.output_dim is not None:
            output = self.output_dense(h)

        if self.return_state:
            log[STATE_KEY] = (h, c)

        if self.return_hidden_state:
            log[HIDDEN_STATE_KEY] = h

        return output, (h, c), log, step_loss
