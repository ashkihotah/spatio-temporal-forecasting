import numpy as np
import keras

from .KerasScaler import KerasScaler

class MinMaxScaler(KerasScaler):
    """Backend-agnostic Min-Max Scaler."""

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._init_min = float(min_val)
        self._init_max = float(max_val)
        self.min_val = self.add_weight(
            name="min_val",
            shape=(),
            initializer=keras.initializers.Constant(self._init_min),
            trainable=False,
            dtype="float32",
        )
        self.max_val = self.add_weight(
            name="max_val",
            shape=(),
            initializer=keras.initializers.Constant(self._init_max),
            trainable=False,
            dtype="float32",
        )

    def normalize(self, x):
        x = keras.ops.convert_to_tensor(x)
        min_v = keras.ops.cast(self.min_val, x.dtype)
        max_v = keras.ops.cast(self.max_val, x.dtype)
        return keras.ops.divide_no_nan(x - min_v, max_v - min_v)

    def denormalize(self, x):
        x = keras.ops.convert_to_tensor(x)
        min_v = keras.ops.cast(self.min_val, x.dtype)
        max_v = keras.ops.cast(self.max_val, x.dtype)
        return (x * (max_v - min_v)) + min_v

    def get_config(self):
        config = super().get_config()
        config.update({
            "min_val": float(keras.ops.convert_to_numpy(self.min_val)),
            "max_val": float(keras.ops.convert_to_numpy(self.max_val)),
        })
        return config

    def to_dict(self) -> dict:
        self_dict = super().to_dict()
        self_dict.update({
            "min_val": float(keras.ops.convert_to_numpy(self.min_val)),
            "max_val": float(keras.ops.convert_to_numpy(self.max_val)),
        })
        return self_dict

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "MinMaxScaler":
        array = np.asarray(array, dtype=np.float64)
        return cls(min_val=float(np.min(array)), max_val=float(np.max(array)))
