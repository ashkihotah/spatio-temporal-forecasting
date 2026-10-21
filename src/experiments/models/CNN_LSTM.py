import logging
import tensorflow as tf
import keras

from experiments.models.seq2seq import AutoRegressiveRNN
from experiments.models import Y_PRED_KEY

logger = logging.getLogger(__name__)

class CNN_LSTM(tf.keras.Model):
    """
    CNN-LSTM encoder-decoder for multi-node time-series forecasting.
    """
    # Idiomatic Keras: Explicitly declare structural dimensions in __init__
    def __init__(
        self, 
        cell_type: str, 
        p: int, n: int, f: int, 
        autocorrelation: bool = False,
        return_hidden_states: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.cell_type = cell_type
        self.p = p
        self.n = n
        self.f = f
        self.autocorrelation = autocorrelation
        self.return_hidden_states = return_hidden_states

        channels = self.n * self.f if self.autocorrelation else self.f
        
        self.cnn = tf.keras.layers.Conv1D(filters=channels, kernel_size=1)
        
        self.enc_dec = AutoRegressiveRNN(
            units=channels,
            p=self.p,
            cell_type=self.cell_type,
            return_hidden_states=return_hidden_states
        )
        self.dense = tf.keras.layers.Dense(self.n)

    def build(self, input_shape):
        channels = self.n * self.f if self.autocorrelation else self.f
        self.cnn.build((input_shape[0], input_shape[1], self.n * self.f))
        self.enc_dec.build((input_shape[0], input_shape[1], channels))
        self.dense.build((input_shape[0], self.p, channels))
        super().build(input_shape)

    def call(self, inputs):
        # Fetch batch size (B) and sequence history (H) dynamically from the current tensor
        input_shape = tf.shape(inputs)
        B = input_shape[0]
        H = input_shape[1]

        # Explicit functional tensor transformation using static n & f, but dynamic B & H
        x = tf.reshape(inputs, (B, H, self.n * self.f)) 
        x = self.cnn(x)  
        
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
            "cell_type": self.cell_type,
            "p": self.p,
            "n": self.n,
            "f": self.f,
            "autocorrelation": self.autocorrelation,
            "return_hidden_states": self.return_hidden_states
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)