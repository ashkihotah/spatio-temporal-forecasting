import tensorflow as tf
import keras

from experiments.models.seq2seq import AutoRegressiveRNN
from experiments.models import Y_PRED_KEY

class LSTM(tf.keras.Model):
    def __init__(
        self, 
        p: int, n: int, f: int, 
        cell_type: str = "LSTM",
        is_bidirectional: bool = False,
        has_attention: bool = False,
        return_hidden_states: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.p = p
        self.n = n
        self.f = f
        self.cell_type = cell_type
        self.is_bidirectional = is_bidirectional
        self.has_attention = has_attention
        self.return_hidden_states = return_hidden_states

        self.enc_dec = AutoRegressiveRNN(
            units=self.n * self.f,
            p=self.p,
            cell_type=self.cell_type,
            has_attention=self.has_attention,
            is_bidirectional=self.is_bidirectional,
            return_hidden_states=self.return_hidden_states
        )
        self.dense = tf.keras.layers.Dense(self.n)

    def build(self, input_shape):
        channels = self.n * self.f
        self.enc_dec.build((input_shape[0], input_shape[1], channels))
        self.dense.build((input_shape[0], self.p, channels))
        super().build(input_shape)

    def call(self, inputs):
        B = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        
        x = tf.reshape(inputs, (B, H, self.n * self.f))
        
        if self.return_hidden_states:
            res_dict = self.enc_dec(x)
            res_dict[Y_PRED_KEY] = self.dense(res_dict[Y_PRED_KEY])
            return res_dict
        else:
            res = self.enc_dec(x)
            return self.dense(res)

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p,
            "n": self.n,
            "f": self.f,
            "cell_type": self.cell_type,
            "is_bidirectional": self.is_bidirectional,
            "has_attention": self.has_attention,
            "return_hidden_states": self.return_hidden_states
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
