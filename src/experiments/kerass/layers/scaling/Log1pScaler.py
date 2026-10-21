import numpy as np
import keras

from .KerasScaler import KerasScaler

class Log1pScaler(KerasScaler):
    """Backend-agnostic logarithmic target scaler."""

    def normalize(self, x):
        x = keras.ops.convert_to_tensor(x)
        return keras.ops.log1p(keras.ops.relu(x))

    def denormalize(self, x):
        x = keras.ops.convert_to_tensor(x)
        return keras.ops.relu(keras.ops.expm1(x))

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "Log1pScaler":
        return cls()
