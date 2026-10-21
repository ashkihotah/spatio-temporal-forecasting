import tensorflow as tf
import keras
import numpy as np
from sklearn.decomposition import IncrementalPCA

from experiments.models.seq2seq import AutoRegressiveRNN
from experiments.models import Y_PRED_KEY

class PCA_LSTM(tf.keras.Model):
    def __init__(
        self, 
        p: int, n: int, f: int, 
        pca_implementation: str = "tf", # "tf" or "sklearn"
        has_dense: bool = False,
        cell_type: str = "LSTM",
        return_hidden_states: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.p = p
        self.n = n
        self.f = f
        self.pca_implementation = pca_implementation
        self.has_dense = has_dense
        self.cell_type = cell_type
        self.return_hidden_states = return_hidden_states

        if self.pca_implementation == "sklearn":
            self.pca = IncrementalPCA(n_components=self.n)
            self.seen_data = []

        self.enc_dec = AutoRegressiveRNN(
            units=self.n,
            p=self.p,
            cell_type=self.cell_type,
            return_hidden_states=self.return_hidden_states
        )
        
        if self.has_dense:
            self.dense = tf.keras.layers.Dense(self.n)

    def build(self, input_shape):
        self.enc_dec.build((input_shape[0], input_shape[1], self.n))
        if self.has_dense:
            self.dense.build((input_shape[0], self.p, self.n))
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p,
            "n": self.n,
            "f": self.f,
            "pca_implementation": self.pca_implementation,
            "has_dense": self.has_dense,
            "cell_type": self.cell_type,
            "return_hidden_states": self.return_hidden_states
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _sklearn_pca_step(self, inputs):
        B = inputs.shape[0]
        H = inputs.shape[1]
        N = self.n
        F = self.f
        
        inputs_reshaped = np.reshape(inputs, (B * H, N * F))
        
        if len(self.seen_data) == 0:
            self.seen_data = inputs_reshaped
        else:
            self.seen_data = np.concatenate((self.seen_data, inputs_reshaped), axis=0)
            
        self.pca.fit(self.seen_data)
        
        transformed = self.pca.transform(inputs_reshaped)
        transformed = np.reshape(transformed, (B, H, N))
        return transformed.astype(np.float32)

    def _tf_pca_step(self, inputs):
        B = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        N = self.n
        F = self.f
        
        inputs_reshaped = tf.reshape(inputs, (B * H, N * F))
        
        mean = tf.reduce_mean(inputs_reshaped, axis=0, keepdims=True)
        centered = inputs_reshaped - mean
        
        s, u, v = tf.linalg.svd(centered)
        v_top = v[:, :N]
        
        transformed = tf.matmul(centered, v_top)
        transformed = tf.reshape(transformed, (B, H, N))
        return transformed

    def call(self, inputs):
        if self.pca_implementation == "sklearn":
            x = tf.numpy_function(self._sklearn_pca_step, [inputs], tf.float32)
            # tf.numpy_function loses shape info, restore it dynamically
            B = tf.shape(inputs)[0]
            H = tf.shape(inputs)[1]
            x.set_shape([None, None, self.n])
            # To ensure compatibility with graph mode dimensions
            x = tf.ensure_shape(x, [None, None, self.n])
        else:
            x = self._tf_pca_step(inputs)
            
        if self.return_hidden_states:
            res_dict = self.enc_dec(x)
            if self.has_dense:
                res_dict[Y_PRED_KEY] = self.dense(res_dict[Y_PRED_KEY])
            return res_dict
        else:
            res = self.enc_dec(x)
            if self.has_dense:
                res = self.dense(res)
            return res
