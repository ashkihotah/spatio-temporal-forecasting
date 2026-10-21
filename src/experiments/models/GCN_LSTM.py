import numpy as np
import tensorflow as tf
import keras

from experiments.models.seq2seq import AutoRegressiveRNN
from experiments.models import Y_PRED_KEY
from experiments.lib.spektral_gcn import GraphConv

class GCN_LSTM(tf.keras.Model):
    def __init__(
        self, 
        p: int, n: int, f: int, 
        adj,
        cell_type: str = "LSTM",
        return_hidden_states: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.p = p
        self.n = n
        self.f = f
        self.adj = adj
        self.cell_type = cell_type
        self.return_hidden_states = return_hidden_states

        self.GCN = GraphConv(self.f, activation="relu")
        
        self.enc_dec = AutoRegressiveRNN(
            units=self.n * self.f,
            p=self.p,
            cell_type=self.cell_type,
            return_hidden_states=self.return_hidden_states
        )
        self.dense = tf.keras.layers.Dense(self.n)

    def build(self, input_shape):
        self.GCN.build([tf.TensorShape([None, self.n, self.f]), tf.TensorShape([self.n, self.n])])
        channels = self.n * self.f
        self.enc_dec.build((input_shape[0], input_shape[1], channels))
        self.dense.build((input_shape[0], self.p, channels))
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
            "p": self.p,
            "n": self.n,
            "f": self.f,
            "adj": self._serialize_adj(self.adj),
            "cell_type": self.cell_type,
            "return_hidden_states": self.return_hidden_states
        })
        return config

    @classmethod
    def from_config(cls, config):
        adj = cls._deserialize_adj(config.pop("adj"))
        return cls(adj=adj, **config)

    def call(self, inputs):
        B = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        
        x = tf.reshape(inputs, (B * H, self.n, self.f))
        
        adj = tf.expand_dims(self.adj, 0)
        adj_batch = tf.tile(adj, [B * H, 1, 1])
        
        x = self.GCN([x, adj_batch])
        x = tf.reshape(x, (B, H, self.n * self.f))
        
        if self.return_hidden_states:
            res_dict = self.enc_dec(x)
            res_dict[Y_PRED_KEY] = self.dense(res_dict[Y_PRED_KEY])
            return res_dict
        else:
            res = self.enc_dec(x)
            return self.dense(res)
