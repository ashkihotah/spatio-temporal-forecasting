import numpy as np
import keras

from experiments.kerass.layers.sequences.constants import (
    DECODER_OUTPUTS_KEY, ENCODER_OUTPUTS_KEY,
)

class Seq2SeqDataset(keras.utils.PyDataset):
    def __init__(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        adjacency: np.ndarray, 
        batch_size: int,
        use_exogenous_features: bool = False,
        shift_targets_in_encoder_inputs: bool = True,
        use_time_features: bool = False,
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.use_exogenous_features = use_exogenous_features
        self.shift_targets_in_encoder_inputs = shift_targets_in_encoder_inputs
        self.use_time_features = use_time_features

        adj_tensor = keras.ops.convert_to_tensor(
            adjacency, 
            dtype=keras.backend.floatx()
        )
        self.adj_tensor = keras.ops.expand_dims(
            adj_tensor, axis=0
        )  # shape (1, N, N)
        self.H = self.x.shape[1]
        self.N = adj_tensor.shape[1]
        self.P = self.y.shape[1]

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def get_encoder_inputs(self, batch_x: np.ndarray):
        in_feats = batch_x[..., 1:] # (B, H, N, F-1)
        last_zero_in_feats = np.zeros_like(in_feats[:, :1, :, :]) # (B, 1, N, F-1)
        shifted_in_feats = np.concatenate(
            (in_feats, last_zero_in_feats), axis=1
        )  # (B, H+1, N, F-1)

        out_feats = batch_x[..., :1] # (B, H, N, 1)
        first_zero_out_feats = np.zeros_like(out_feats[:, :1, :, :]) # (B, 1, N, 1)
        shifted_out_feats = np.concatenate(
            (first_zero_out_feats, out_feats), axis=1
        )  # (B, H+1, N, 1)

        encoder_inputs = np.concatenate(
            (shifted_out_feats, shifted_in_feats), axis=-1
        )  # (B, H+1, N, F)

        return encoder_inputs
    
    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, self.x.shape[0])
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]

        if not self.use_time_features:
            batch_x = batch_x[..., :-6]  # (B, H, N, F-6)
            batch_y = batch_y[..., :-6]  # (B, P, N, F-6)

        enc_adj_tensor = keras.ops.broadcast_to(
            self.adj_tensor, 
            (batch_x.shape[0], self.H, self.N, self.N)
        )  # shape (B, H, N, N)

        dec_adj_tensor = keras.ops.broadcast_to(
            self.adj_tensor, 
            (batch_x.shape[0], self.P, self.N, self.N)
        )  # shape (B, P, N, N)

        if self.use_exogenous_features:
            decoder_y_features = batch_y[..., 1:]
        else:
            decoder_y_features = np.empty(
                (batch_x.shape[0], self.P, self.N, 0),
                dtype=batch_x.dtype,
            )

        if self.shift_targets_in_encoder_inputs:
            encoder_inputs = self.get_encoder_inputs(batch_x)[..., :-1, :, :]  # (B, H, N, F)
        else:
            encoder_inputs = batch_x # (B, H, N, F)

        result = (
            (
                ( # encoder_inputs
                    encoder_inputs,
                    enc_adj_tensor
                ),
                ( # decoder_inputs
                    decoder_y_features,
                    dec_adj_tensor,
                ),
                # encoder_time_features
                # decoder_time_features

                # decoder_initial_outputs
                batch_x[..., -1, :, :1], # (B, N, 1)
            ),
            {
                ENCODER_OUTPUTS_KEY: batch_x[..., :1], # (B, H, N, 1)
                DECODER_OUTPUTS_KEY: batch_y[..., :1], # (B, P, N, 1)
            }
        )

        return result

    def get_shapes(self):
        batch = self.__getitem__(0)
        shapes = keras.tree.map_structure(
            lambda x: x.shape[1:], batch
        )
        return shapes
