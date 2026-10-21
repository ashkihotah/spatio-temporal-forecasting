import numpy as np
import scipy
import tensorflow as tf
import keras

from experiments.lib.spektral_gcn import GraphConv
from experiments.models import Y_PRED_KEY, ENC_STATES_KEY

class GCLSTM(tf.keras.Model):
    def __init__(
        self, 
        h: int, p: int, n: int, f: int, 
        adj,
        node: int = 0,
        ablation: str = None,
        return_hidden_states: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.h = h
        self.p = p
        self.n = n
        self.f = f
        self.adj = adj
        self.node = node
        self.ablation = ablation
        self.return_hidden_states = return_hidden_states

        self.gcn1 = GraphConv(32, activation="relu", name="gcn1")
        self.gcn2 = GraphConv(16, activation="relu", name="gcn2")

        self.conv1 = tf.keras.layers.Conv1D(
            4, 3, padding="same", data_format="channels_last", name="conv1"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            4, 3, padding="same", data_format="channels_last", name="conv2"
        )

        self.bn = tf.keras.layers.BatchNormalization()

        self.pooling = tf.keras.layers.AveragePooling1D(
            2, padding="same", data_format="channels_last", name="pooling"
        )

        self.lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, name="lstm1")
        self.lstm2 = tf.keras.layers.LSTM(128, name="lstm2")

        self.dropout = tf.keras.layers.Dropout(0.5, name="dropout")
        self.out = tf.keras.layers.Dense(self.p, name="out")

    def build(self, input_shape):
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
        config.update({
            "h": self.h,
            "p": self.p,
            "n": self.n,
            "f": self.f,
            "adj": self._serialize_adj(self.adj),
            "node": self.node,
            "ablation": self.ablation,
            "return_hidden_states": self.return_hidden_states
        })
        return config

    @classmethod
    def from_config(cls, config):
        adj = cls._deserialize_adj(config.pop("adj"))
        return cls(adj=adj, **config)

    @staticmethod
    def compute_node_features(inputs, H, N):
        B = tf.shape(inputs)[0]
        features = tf.stack(
            [
                tf.math.reduce_mean(inputs[..., 0], axis=1),
                tf.math.reduce_mean(inputs[:, H // 2 :, :, 0], axis=1),
                tf.math.reduce_std(inputs[..., 0], axis=1),
                tf.math.reduce_std(inputs[:, H // 2 :, :, 0], axis=1),
                tf.numpy_function(
                    lambda x: np.nan_to_num(
                        np.array(
                            [
                                [
                                    scipy.stats.skew(
                                        x[b, :, n, 0], axis=None, nan_policy="raise"
                                    )
                                    for n in range(N)
                                ]
                                for b in range(x.shape[0])
                            ],
                            dtype=np.float32,
                        )
                    ),
                    [inputs],
                    tf.float32,
                ),
                tf.numpy_function(
                    lambda x: np.nan_to_num(
                        np.array(
                            [
                                [
                                    scipy.stats.kurtosis(
                                        x[b, :, n, 0], axis=None, nan_policy="raise"
                                    )
                                    for n in range(N)
                                ]
                                for b in range(x.shape[0])
                            ],
                            dtype=np.float32,
                        )
                    ),
                    [inputs],
                    tf.float32,
                ),
                tf.numpy_function(
                    lambda x: np.nan_to_num(
                        np.array(
                            [
                                [
                                    np.polyfit(np.arange(H), x[b, :, n, 0], 1)[0]
                                    for n in range(N)
                                ]
                                for b in range(x.shape[0])
                            ],
                            dtype=np.float32,
                        )
                    ),
                    [inputs],
                    tf.float32,
                ),
            ],
            axis=2,
        )
        features.set_shape([None, N, 7])
        return features

    def call(self, inputs):
        B = tf.shape(inputs)[0]
        H = self.h
        N = self.n

        node_features = self.compute_node_features(inputs, H, N)
        if self.ablation == "nosummary":
            node_features = node_features[..., :1]

        adj_batch = tf.tile(tf.expand_dims(self.adj, 0), [B, 1, 1])
        
        gcn_out = self.gcn1([node_features, adj_batch])
        gcn_out = self.gcn2([gcn_out, adj_batch])

        noconv_gcn_out = tf.identity(gcn_out)
        noconv_gcn_out = tf.keras.layers.Flatten(data_format="channels_last")(noconv_gcn_out)

        # Transpose [B, N, 16] to [B, 16, N] to use channels_last properly
        gcn_out = tf.transpose(gcn_out, perm=[0, 2, 1])
        
        gcn_out = self.conv1(gcn_out)
        gcn_out = self.pooling(gcn_out)
        first_conv_out = tf.identity(gcn_out)
        gcn_out = tf.nn.relu(gcn_out)
        gcn_out = self.conv2(gcn_out)

        if not self.ablation == "noskip":
            gcn_out = gcn_out + first_conv_out
            gcn_out = self.bn(gcn_out)

        # Flatten the features while keeping the batch dimension
        gcn_out = tf.keras.layers.Flatten()(gcn_out)

        lstm_input = inputs[:, :, self.node, :]
        lstm_states = self.lstm1(lstm_input)
        lstm_out = self.lstm2(lstm_states)

        if not self.ablation == "noconv":
            out_features = tf.concat((gcn_out, lstm_out), axis=-1)
        else:
            out_features = tf.concat((noconv_gcn_out, lstm_out), axis=-1)

        out_features = self.dropout(out_features)
        y = self.out(out_features)

        if self.return_hidden_states:
            res_dict = {
                Y_PRED_KEY: y,
                ENC_STATES_KEY: lstm_states
            }
            return res_dict
        else:
            return y


class GCLSTM_OneClass(tf.keras.Model):
    def __init__(self, h: int, p: int, n: int, adj=None, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        self.p = p
        self.n = n
        self.adj = adj
        self.node = 0
        self.real_f = [31, 8, 3, 9]

        self.mappers = [
            tf.keras.layers.Dense(32, activation="relu")
            for _ in range(self.n)
        ]
        self.encoder = tf.keras.layers.Dense(8, activation="relu")
        self.decoder = tf.keras.layers.Dense(32, activation="relu")
        self.demappers = [
            tf.keras.layers.Dense(self.real_f[node], activation="relu")
            for node in range(self.n)
        ]

    def build(self, input_shape):
        super().build(input_shape)

    @staticmethod
    def _serialize_adj(adj):
        if isinstance(adj, tf.Tensor):
            adj = adj.numpy()
        if isinstance(adj, np.ndarray):
            return {"type": "ndarray", "value": adj.tolist(), "dtype": str(adj.dtype)}
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
        config.update({
            "h": self.h, "p": self.p, "n": self.n,
            "adj": self._serialize_adj(self.adj)
        })
        return config

    @classmethod
    def from_config(cls, config):
        adj = cls._deserialize_adj(config.pop("adj"))
        return cls(adj=adj, **config)

    def call(self, inputs):
        B = tf.shape(inputs)[0]
        F = tf.shape(inputs)[3]

        common_space_mapping = [
            self.mappers[n](inputs[:, :, n, : self.real_f[n]]) for n in range(min(self.n, len(self.real_f)))
        ]
        common_space_mapping = tf.stack(common_space_mapping, axis=2)

        encoded = self.encoder(common_space_mapping)
        decoded = self.decoder(encoded)

        demapped = [self.demappers[n](decoded[:, :, n]) for n in range(min(self.n, len(self.real_f)))]
        demapped = [
            tf.pad(d, paddings=[[0, 0], [0, 0], [0, F - tf.shape(d)[-1]]]) for d in demapped
        ]
        demapped = tf.stack(demapped, axis=2)

        return demapped
