import numpy as np
import tensorflow as tf
import keras

from experiments.lib.spektral_utilities import *
from experiments.lib.spektral_gcn import GraphConv

from experiments.models import (
    Y_PRED_KEY,
    ENC_STATES_KEY,
    DEC_STATES_KEY,
    ATT_WEIGHTS_KEY,
)


class GAP_LSTM(tf.keras.Model):
    """
    Graph Attention Pyramidal LSTM for multi-node time-series forecasting.

    Idiomatic Keras: the structural dimensions ``p`` (prediction steps),
    ``n`` (number of nodes) and ``f`` (number of node features) are explicit
    ``__init__`` arguments, so every sub-layer can be fully constructed up
    front. Only the history length ``h`` is inferred dynamically from the
    input tensor at call time.

    When ``return_interpretations`` is True the ``call`` returns, alongside the
    prediction, the encoder/decoder hidden states and the attention weights so
    that an external caller can compute Moran's I spatial autocorrelation or
    plot the learned representations.
    """

    def __init__(
        self,
        p: int,
        n: int,
        f: int,
        adj,
        cell_type: str = "gclstm",
        conv: bool = True,
        attention: bool = True,
        gnn_type: str = "spektral",  # spektral, weighted, gat
        return_hidden_states: bool = False,
        return_attention_weights: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Arguments:
        p: prediction steps (target forecasting length)
        n: number of nodes
        f: number of node features
        adj: adjacency matrix

        cell_type: type of the recurrent cell. One of ``gclstm``,
            ``nomemorystate`` (default: ``gclstm``)
        conv: whether the 2D convolution branch is active (default: True)
        attention: whether the attention mechanism is active (default: True)
        gnn_type: graph convolution type. One of ``spektral``, ``weighted``,
            ``gat`` (default: ``spektral``)
        return_interpretations: when True, ``call`` additionally returns the
            encoder/decoder hidden states and attention weights (default: False)
        """
        super().__init__(**kwargs)

        self.p = p
        self.n = n
        self.f = f
        self.adj = adj
        self.cell_type = cell_type
        self.has_conv = conv
        self.has_attention = attention
        self.gnn_type = gnn_type
        self.verbose = verbose
        self.return_hidden_states = return_hidden_states
        self.return_attention_weights = return_attention_weights

        self.initial_norm = tf.keras.layers.LayerNormalization()
        self.start_ff = FeedForward([self.f])
        self.sagl = SA_GCLSTM(
            p=self.p,
            n=self.n,
            f=self.f,
            cell_type=self.cell_type,
            has_conv=self.has_conv,
            has_attention=self.has_attention,
            gnn_type=self.gnn_type,
            verbose=self.verbose,
        )
        self.end_ff = FeedForward([1])

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        
        self.initial_norm.build(input_shape)
        self.start_ff.build(input_shape)
        
        adj_shape = tf.TensorShape([self.n, self.n])
        self.sagl.build([input_shape, adj_shape])
        
        # Output shape of SAGL is (batch, p, n, f)
        dec_shape = tf.TensorShape([input_shape[0], self.p, self.n, self.f])
        self.end_ff.build(dec_shape)
        
        super().build(input_shape)

    @staticmethod
    def _serialize_adj(adj):
        if isinstance(adj, tf.Tensor):
            adj = adj.numpy()
        if isinstance(adj, np.ndarray):
            return {
                "type": "ndarray",
                "value": adj.tolist(),
                "dtype": str(adj.dtype),
            }
        return adj

    @staticmethod
    def _deserialize_adj(adj_data):
        if isinstance(adj_data, dict) and adj_data.get("type") == "ndarray":
            return np.array(adj_data.get("value"), dtype=adj_data.get("dtype"))
        if isinstance(adj_data, list):
            return np.array(adj_data)
        return adj_data

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "p": self.p,
                "n": self.n,
                "f": self.f,
                "adj": self._serialize_adj(self.adj),
                "cell_type": self.cell_type,
                "conv": self.has_conv,
                "attention": self.has_attention,
                "gnn_type": self.gnn_type,
                "return_hidden_states": self.return_hidden_states,
                "return_attention_weights": self.return_attention_weights,
                "verbose": self.verbose,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        adj = cls._deserialize_adj(config.pop("adj"))
        return cls(adj=adj, **config)

    def call(self, inputs):
        x = inputs  # (b, h, n, f)
        adj = self.adj

        x = self.initial_norm(x)
        x = self.start_ff(x)
        output, enc_hidden_states, dec_hidden_states, attention_weights = self.sagl(
            [x, adj]
        )
        y = self.end_ff(output)  # (b, p, n, 1)
        y = tf.squeeze(y, axis=-1)  # (b, p, n)

        if self.return_hidden_states or self.return_attention_weights:
            result = {Y_PRED_KEY: y}
            if self.return_hidden_states:
                result[ENC_STATES_KEY] = enc_hidden_states
                result[DEC_STATES_KEY] = dec_hidden_states
            if self.return_attention_weights and attention_weights is not None:
                    result[ATT_WEIGHTS_KEY] = attention_weights
            return result
        else:
            return y


class SA_GCLSTM(tf.keras.layers.Layer):
    """Spatial-Attention Graph-Convolutional LSTM encoder-decoder block."""

    def __init__(
        self,
        p: int,
        n: int,
        f: int,
        cell_type: str = "gclstm",
        has_conv: bool = True,
        has_attention: bool = True,
        gnn_type: str = "spektral",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p = p
        self.n = n
        self.f = f
        self.cell_type = cell_type
        self.has_conv = has_conv
        self.has_attention = has_attention
        self.gnn_type = gnn_type
        self.verbose = verbose

        self.encoder = Encoder(
            n=self.n,
            f=self.f,
            gnn_type=self.gnn_type,
            cell_type=self.cell_type,
            verbose=self.verbose,
        )
        self.decoder = Decoder(
            p=self.p,
            n=self.n,
            f=self.f,
            gnn_type=self.gnn_type,
            cell_type=self.cell_type,
            has_attention=self.has_attention,
            verbose=self.verbose,
        )
        self.conv = (
            tf.keras.layers.Conv2D(
                self.p, (2, 2), data_format="channels_last", padding="same"
            )
            if self.has_conv
            else None
        )

    def build(self, input_shape):
        x_shape, adj_shape = input_shape
        x_shape = tf.TensorShape(x_shape)
        adj_shape = tf.TensorShape(adj_shape)
        
        # Propagate to Encoder
        self.encoder.build([x_shape, adj_shape])
        
        # Propagate to Decoder
        enc_last_shape = tf.TensorShape([x_shape[0], self.n, self.f])
        hidden_states_shape = tf.TensorShape([x_shape[0], x_shape[1], self.n, self.f])
        self.decoder.build(
            [
                enc_last_shape,
                enc_last_shape,
                enc_last_shape,
                hidden_states_shape,
                adj_shape,
            ]
        )
        
        # Propagate to 2D Convolution (if active)
        if self.has_conv:
            conv_input_shape = tf.TensorShape([x_shape[0], self.n, self.f, 1])
            self.conv.build(conv_input_shape)
            
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "p": self.p,
                "n": self.n,
                "f": self.f,
                "cell_type": self.cell_type,
                "has_conv": self.has_conv,
                "has_attention": self.has_attention,
                "gnn_type": self.gnn_type,
                "verbose": self.verbose,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        x, adj = inputs

        enc_hidden_states, enc_last_h, enc_last_c, enc_last_m = self.encoder([x, adj])
        dec_hidden_states, attention_weights = self.decoder(
            [enc_last_h, enc_last_c, enc_last_m, enc_hidden_states, adj]
        )

        output = dec_hidden_states
        if self.has_conv:
            conv_input = tf.expand_dims(enc_last_h, axis=-1)  # (b, n, f, 1)
            conv_output = self.conv(conv_input)  # (b, n, f, p)
            conv_output = tf.transpose(conv_output, perm=[0, 3, 1, 2])  # (b, p, n, f)
            output = dec_hidden_states + conv_output

        return output, enc_hidden_states, dec_hidden_states, attention_weights


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n, f, gnn_type, cell_type="gclstm", verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.f = f
        self.gnn_type = gnn_type
        self.cell_type = cell_type
        self.verbose = verbose

        if self.cell_type == "gclstm":
            self.cell = GCNLSTM_Cell(self.gnn_type, n_features=self.f)
        elif self.cell_type == "nomemorystate":
            self.cell = GCNLSTM_Cell(
                self.gnn_type, n_features=self.f, has_memory_state=False
            )
        else:
            raise NotImplementedError(
                f"Unsupported encoder cell_type: {self.cell_type}"
            )

    def call(self, inputs):
        x, adj = inputs  # x: (b, h, n, f)
        b = tf.shape(x)[0]
        h = x.shape[1]  # history length is static

        last_h = tf.ones((b, self.n, self.f))
        last_c = tf.ones((b, self.n, self.f))
        last_m = tf.ones((b, self.n, self.f))

        hidden_states = []
        for i in range(h):
            last_h, last_c, last_m = self.cell(
                [x[:, i, :, :], last_h, last_c, last_m, adj]
            )
            hidden_states.append(last_h)

        hidden_states = tf.stack(hidden_states, axis=1)  # (b, h, n, f)
        return hidden_states, last_h, last_c, last_m

    def build(self, input_shape):
        x_shape, adj_shape = input_shape
        x_shape = tf.TensorShape(x_shape)
        adj_shape = tf.TensorShape(adj_shape)
        
        # Build GCNLSTM Cell with expected step input shapes
        step_shape = tf.TensorShape([x_shape[0], self.n, self.f])
        self.cell.build(
            [step_shape, step_shape, step_shape, step_shape, adj_shape]
        )
        
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n": self.n,
                "f": self.f,
                "gnn_type": self.gnn_type,
                "cell_type": self.cell_type,
                "verbose": self.verbose,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiheadAttention(tf.keras.layers.Layer):
    """
    Per-node multihead attention. The number of nodes ``n`` and features ``f``
    are explicit ``__init__`` arguments; only the number of attention keys
    ``h`` (the encoder history length) is inferred dynamically in ``build``.
    """

    def __init__(self, n, f, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.f = f
        self.verbose = verbose

        # Query projections depend only on the number of nodes / features.
        self.Wq = [FeedForward([self.f]) for _ in range(self.n)]
        # Key/value projections depend on the history length and are built lazily.
        self._Wk_flat = []
        self._Wv_flat = []

    def call(self, inputs):
        Q, K, V = inputs

        heads = []
        log_scores = []
        for node in range(self.n):
            Qn = Q[:, node, :]
            Kn = K[:, :, node, :]
            Vn = V[:, :, node, :]

            QWq = self.Wq[node](Qn)

            KWk = tf.stack([self.Wk[node][i](Kn[:, i]) for i in range(self.h)])
            VWv = tf.stack([self.Wv[node][i](Vn[:, i]) for i in range(self.h)])

            KWk = tf.transpose(KWk, perm=[1, 2, 0])
            VWv = tf.transpose(VWv, perm=[1, 0, 2])

            scores = tf.nn.softmax(
                tf.linalg.matmul(tf.expand_dims(QWq, axis=1), KWk)
            )
            log_scores.append(scores)

            result = tf.linalg.matmul(scores, VWv)
            result = tf.squeeze(result, axis=[1])
            heads.append(result)

        log_scores = tf.stack(log_scores, axis=1)
        log_scores = tf.squeeze(log_scores, axis=2)

        heads = tf.stack(heads)
        heads = tf.transpose(heads, perm=[1, 0, 2])
        return heads, log_scores

    def build(self, input_shape):
        q_shape, k_shape, _ = input_shape
        q_shape = tf.TensorShape(q_shape)
        k_shape = tf.TensorShape(k_shape)
        
        self.h = int(k_shape[1])
        
        # 1. Build Query Projections
        for node in range(self.n):
            self.Wq[node].build(tf.TensorShape([q_shape[0], self.f]))
            
        # 2. Instantiate inner lists
        self.Wk = [
            [FeedForward([self.f]) for _ in range(self.h)] for _ in range(self.n)
        ]
        self.Wv = [
            [FeedForward([self.f]) for _ in range(self.h)] for _ in range(self.n)
        ]
        
        # 3. Explicitly build all projection layers
        for node in range(self.n):
            for step in range(self.h):
                self.Wk[node][step].build(tf.TensorShape([q_shape[0], self.f]))
                self.Wv[node][step].build(tf.TensorShape([q_shape[0], self.f]))
                
        super().build(input_shape)

    # def build(self, input_shape):
    #     _, k_shape, _ = input_shape
    #     self.h = int(k_shape[1])
        
    #     # Instantiate and build layers
    #     self._Wk_flat = []
    #     self._Wv_flat = []
    #     for _ in range(self.n * self.h):
    #         wk = FeedForward([self.f])
    #         wv = FeedForward([self.f])
    #         wk.build([None, self.f])
    #         wv.build([None, self.f])
    #         self._Wk_flat.append(wk)
    #         self._Wv_flat.append(wv)
            
    #     # Reconstruct structured lookups for call logic
    #     self.Wk = [self._Wk_flat[i*self.h : (i+1)*self.h] for i in range(self.n)]
    #     self.Wv = [self._Wv_flat[i*self.h : (i+1)*self.h] for i in range(self.n)]
        
    #     super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"n": self.n, "f": self.f, "verbose": self.verbose})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        p,
        n,
        f,
        gnn_type,
        cell_type="gclstm",
        has_attention=True,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p = p
        self.n = n
        self.f = f
        self.gnn_type = gnn_type
        self.cell_type = cell_type
        self.has_attention = has_attention
        self.verbose = verbose

        if self.cell_type == "gclstm":
            self.cell = GCNLSTM_Cell(self.gnn_type, n_features=self.f)
        elif self.cell_type == "nomemorystate":
            self.cell = GCNLSTM_Cell(
                self.gnn_type, n_features=self.f, has_memory_state=False
            )
        elif self.cell_type == "lstm":
            self.cell = tf.keras.layers.LSTMCell(self.n * self.f)
        else:
            raise NotImplementedError(
                f"Unsupported decoder cell_type: {self.cell_type}"
            )

        self.attention = (
            MultiheadAttention(n=self.n, f=self.f, verbose=self.verbose)
            if self.has_attention
            else None
        )

    def call(self, inputs):
        (
            last_hidden_state,
            last_cell_state,
            last_memory_state,
            hidden_states,
            adj,
        ) = inputs
        b = tf.shape(hidden_states)[0]
        h = hidden_states.shape[1]

        dec_result = []
        batch_scores = []
        for i in range(self.p):
            if self.has_attention:
                if isinstance(self.cell, tf.keras.layers.LSTMCell):
                    last_hidden_state = tf.reshape(last_hidden_state, [b, 1, -1])
                    hidden_states = tf.reshape(hidden_states, [b, h, 1, -1])
                decoder_input, scores = self.attention(
                    [last_hidden_state, hidden_states, hidden_states]
                )
                batch_scores.append(scores)
            else:
                decoder_input = last_hidden_state

            if isinstance(self.cell, tf.keras.layers.LSTMCell):
                next_hidden_state, [_, next_cell_state] = self.cell(
                    tf.reshape(decoder_input, [b, -1]),
                    [
                        tf.reshape(last_hidden_state, [b, -1]),
                        tf.reshape(last_cell_state, [b, -1]),
                    ],
                )
                next_memory_state = tf.zeros_like(next_hidden_state)
            else:
                next_hidden_state, next_cell_state, next_memory_state = self.cell(
                    [
                        decoder_input,
                        last_hidden_state,
                        last_cell_state,
                        last_memory_state,
                        adj,
                    ]
                )

            last_hidden_state = next_hidden_state
            last_cell_state = next_cell_state
            last_memory_state = next_memory_state

            dec_result.append(next_hidden_state)

        dec_output = tf.stack(dec_result, axis=1)  # (b, p, n, f)
        attention_weights = None
        if self.has_attention:
            attention_weights = tf.stack(batch_scores, axis=1)  # (b, p, n, h)

        return dec_output, attention_weights

    def build(self, input_shape):
        (
            last_hidden_shape,
            last_cell_shape,
            last_memory_shape,
            hidden_states_shape,
            adj_shape,
        ) = input_shape
        
        last_hidden_shape = tf.TensorShape(last_hidden_shape)
        hidden_states_shape = tf.TensorShape(hidden_states_shape)
        adj_shape = tf.TensorShape(adj_shape)
        
        # Propagate to Multihead Attention
        if self.has_attention:
            self.attention.build(
                [
                    last_hidden_shape,
                    hidden_states_shape,
                    hidden_states_shape
                ]
            )
            
        # Build Cell
        step_shape = last_hidden_shape
        if not isinstance(self.cell, tf.keras.layers.LSTMCell):
            self.cell.build([step_shape, step_shape, step_shape, step_shape, adj_shape])
            
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "p": self.p,
                "n": self.n,
                "f": self.f,
                "gnn_type": self.gnn_type,
                "cell_type": self.cell_type,
                "has_attention": self.has_attention,
                "verbose": self.verbose,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GCNLSTM_Cell(tf.keras.layers.Layer):
    def __init__(
        self,
        gnn_type,
        n_features,
        has_memory_state=True,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.verbose = verbose
        self.gnn_type = gnn_type
        self.n_features = n_features
        self.has_memory_state = has_memory_state

        gnn_layer_class = {
            "spektral": GraphConv,
            "weighted": WeightedGraphConv,
            "gat": GAT,
        }.get(gnn_type, GraphConv)

        self.fx_gnn = gnn_layer_class(self.n_features, activation="relu")
        self.fh_gnn = gnn_layer_class(self.n_features, activation="relu")
        self.ix_gnn = gnn_layer_class(self.n_features, activation="relu")
        self.ih_gnn = gnn_layer_class(self.n_features, activation="relu")
        self.cx_gnn = gnn_layer_class(self.n_features, activation="relu")
        self.ch_gnn = gnn_layer_class(self.n_features, activation="relu")
        self.ox_gnn = gnn_layer_class(self.n_features, activation="relu")
        self.oh_gnn = gnn_layer_class(self.n_features, activation="relu")

        if self.has_memory_state:
            self.i_gnn = gnn_layer_class(self.n_features, activation="relu")
            self.g_gnn = gnn_layer_class(self.n_features, activation="relu")
            self.o_gnn = gnn_layer_class(self.n_features, activation="relu")
            self.im_gnn = gnn_layer_class(self.n_features, activation="relu")
            self.gm_gnn = gnn_layer_class(self.n_features, activation="relu")
            self.om_gnn = gnn_layer_class(self.n_features, activation="relu")

    def build(self, input_shape):
        x_shape, _, _, _, adj_shape = input_shape
        x_shape = tf.TensorShape(x_shape)
        adj_shape = tf.TensorShape(adj_shape)
        gnn_input = [x_shape, adj_shape]

        self.fx_gnn.build(gnn_input)
        self.fh_gnn.build(gnn_input)
        self.ix_gnn.build(gnn_input)
        self.ih_gnn.build(gnn_input)
        self.cx_gnn.build(gnn_input)
        self.ch_gnn.build(gnn_input)
        self.ox_gnn.build(gnn_input)
        self.oh_gnn.build(gnn_input)

        if self.has_memory_state:
            self.i_gnn.build(gnn_input)
            self.g_gnn.build(gnn_input)
            self.o_gnn.build(gnn_input)
            self.im_gnn.build(gnn_input)
            self.gm_gnn.build(gnn_input)
            self.om_gnn.build(gnn_input)

        super().build(input_shape)

    def call(self, inputs):
        x, hidden_state, cell_state, memory_state, adj = inputs

        fx = self.fx_gnn([x, adj])
        fh = self.fh_gnn([hidden_state, adj])
        ix = self.ix_gnn([x, adj])
        ih = self.ih_gnn([hidden_state, adj])
        cx = self.cx_gnn([x, adj])
        ch = self.ch_gnn([hidden_state, adj])
        ox = self.ox_gnn([x, adj])
        oh = self.oh_gnn([hidden_state, adj])

        f = tf.math.sigmoid(fx + fh)
        i = tf.math.sigmoid(ix + ih)
        o = tf.math.sigmoid(ox + oh)
        c = f * cell_state + i * tf.math.tanh(cx + ch)
        h = o * tf.math.tanh(c)

        if self.has_memory_state:
            SA_ih = self.i_gnn([h, adj])
            SA_im = self.im_gnn([memory_state, adj])
            SA_gh = self.g_gnn([h, adj])
            SA_gm = self.gm_gnn([memory_state, adj])
            SA_oh = self.o_gnn([h, adj])
            SA_om = self.om_gnn([memory_state, adj])

            i = tf.math.sigmoid(SA_ih + SA_im)
            g = tf.math.sigmoid(SA_gh + SA_gm)
            o = tf.math.sigmoid(SA_oh + SA_om)

            m = i * memory_state + (1 - i) * g
            h = m * o

        return h, c, m if self.has_memory_state else tf.zeros_like(h)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "gnn_type": self.gnn_type,
                "n_features": self.n_features,
                "has_memory_state": self.has_memory_state,
                "verbose": self.verbose,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class WeightedGraphConv(tf.keras.layers.Layer):
    def __init__(self, dim, verbose=False, **kwargs):
        super().__init__(**kwargs)

        self.verbose = verbose
        self.dim = dim
        self.dense = FeedForward([2 * dim])

    def call(self, inputs):
        x, adj = inputs
        x = tf.matmul(adj, x)
        out = self.dense(x)
        ls, rs = tf.split(out, 2, axis=-1)
        out = ls * tf.math.sigmoid(rs)
        return out

    def build(self, input_shape):
        x_shape, _ = input_shape
        x_shape = tf.TensorShape(x_shape)
        
        self.dense.build(x_shape)
        
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "verbose": self.verbose})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GAT(tf.keras.layers.Layer):
    def __init__(self, dim, verbose=False, **kwargs):
        super().__init__(**kwargs)

        self.verbose = verbose
        self.dim = dim

        self.Wq = tf.keras.layers.Dense(self.dim)
        self.Wk = tf.keras.layers.Dense(self.dim)
        self.Wv = tf.keras.layers.Dense(self.dim)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        x, _ = inputs

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = tf.linalg.matmul(Q, tf.transpose(K, (0, 2, 1)))
        scores = tf.math.divide(scores, tf.math.sqrt(tf.cast(self.dim, tf.float32)))

        adj = tf.ones_like(scores)
        scores = self.softmax(scores, mask=adj)

        result = tf.linalg.matmul(scores, V)
        return result

    def build(self, input_shape):
        x_shape, _ = input_shape
        x_shape = tf.TensorShape(x_shape)
        
        self.Wq.build(x_shape)
        self.Wk.build(x_shape)
        self.Wv.build(x_shape)
        
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "verbose": self.verbose})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, sizes, normalize=False, verbose=False, **kwargs):
        super().__init__(**kwargs)

        self.verbose = verbose
        self.sizes = list(sizes)
        self.normalize = normalize
        self.dense_layers = [tf.keras.layers.Dense(units=size) for size in sizes]
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, center=False, scale=False
        )

    def build(self, input_shape):
        shape = input_shape
        for layer in self.dense_layers:
            layer.build(shape)
            shape = list(shape)
            shape[-1] = layer.units
        if self.normalize:
            self.layer_norm.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        x = inputs
        for i, layer in enumerate(self.dense_layers):
            x = layer(x)
            if i < len(self.dense_layers) - 1:
                x = tf.nn.relu(x)
        if self.normalize:
            x += inputs
            x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sizes": self.sizes,
                "normalize": self.normalize,
                "verbose": self.verbose,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
