import numpy as np
import keras

from .KerasScaler import KerasScaler

class IdentityScaler(KerasScaler):
    """Pass-through scaler for features that do not require scaling."""

    def normalize(self, x):
        return keras.ops.convert_to_tensor(x)

    def denormalize(self, x):
        return keras.ops.convert_to_tensor(x)

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "IdentityScaler":
        return cls()
