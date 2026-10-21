import tensorflow as tf
import keras

from experiments.models.seq2seq import AutoRegressiveRNN
from experiments.models import Y_PRED_KEY

class SVD_LSTM(tf.keras.Model):
    def __init__(
        self, 
        p: int, n: int, f: int, 
        order: int = 3,
        cell_type: str = "LSTM",
        return_hidden_states: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.p = p
        self.n = n
        self.f = f
        self.order = order
        self.cell_type = cell_type
        self.return_hidden_states = return_hidden_states

        self.dense1 = tf.keras.layers.Dense(self.n * self.f)
        
        self.enc_dec = AutoRegressiveRNN(
            units=self.n * self.f,
            p=self.p,
            cell_type=self.cell_type,
            return_hidden_states=self.return_hidden_states
        )
        self.dense2 = tf.keras.layers.Dense(self.n)

    def build(self, input_shape):
        channels = self.n * self.f
        self.dense1.build((input_shape[0], input_shape[1], channels))
        self.enc_dec.build((input_shape[0], input_shape[1], channels))
        self.dense2.build((input_shape[0], self.p, channels))
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p,
            "n": self.n,
            "f": self.f,
            "order": self.order,
            "cell_type": self.cell_type,
            "return_hidden_states": self.return_hidden_states
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        B = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        
        s, u, v = tf.linalg.svd(inputs)
        
        Z = tf.shape(s)[-1]
        order = tf.minimum(self.order, Z)
        mask = tf.concat([
            tf.ones(order, dtype=s.dtype), 
            tf.zeros(Z - order, dtype=s.dtype)
        ], axis=0)
        s = s * mask
        
        s_diag = tf.linalg.diag(s)
        inputs_approx = tf.linalg.matmul(u, s_diag)
        inputs_approx = tf.linalg.matmul(inputs_approx, v, transpose_b=True)
        inputs_approx = tf.reshape(inputs_approx, [B, H, self.n * self.f])
        
        x = self.dense1(inputs_approx)
        
        if self.return_hidden_states:
            res_dict = self.enc_dec(x)
            res_dict[Y_PRED_KEY] = self.dense2(res_dict[Y_PRED_KEY])
            return res_dict
        else:
            res = self.enc_dec(x)
            return self.dense2(res)
