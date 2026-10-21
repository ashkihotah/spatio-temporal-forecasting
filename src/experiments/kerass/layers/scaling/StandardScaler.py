import numpy as np
import keras

from .KerasScaler import KerasScaler

class StandardScaler(KerasScaler):
    """Backend-agnostic equivalent of sklearn.preprocessing.StandardScaler."""

    def __init__(self, mean: float = 0.0, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._init_mean = float(mean)
        self._init_scale = float(scale)
        self.mean = self.add_weight(
            name="mean",
            shape=(),
            initializer=keras.initializers.Constant(self._init_mean),
            trainable=False,
            dtype="float32",
        )
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=keras.initializers.Constant(self._init_scale),
            trainable=False,
            dtype="float32",
        )

    def normalize(self, x):
        x = keras.ops.convert_to_tensor(x)
        mean = keras.ops.cast(self.mean, x.dtype)
        scale = keras.ops.cast(self.scale, x.dtype)
        return keras.ops.divide_no_nan(x - mean, scale)

    def denormalize(self, x):
        x = keras.ops.convert_to_tensor(x)
        mean = keras.ops.cast(self.mean, x.dtype)
        scale = keras.ops.cast(self.scale, x.dtype)
        return (x * scale) + mean

    def get_config(self):
        config = super().get_config()
        config.update({
            "mean": float(keras.ops.convert_to_numpy(self.mean)),
            "scale": float(keras.ops.convert_to_numpy(self.scale)),
        })
        return config

    def to_dict(self) -> dict:
        self_dict = super().to_dict()
        self_dict.update({
            "mean": float(keras.ops.convert_to_numpy(self.mean)),
            "scale": float(keras.ops.convert_to_numpy(self.scale)),
        })
        return self_dict

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "StandardScaler":
        array = np.asarray(array, dtype=np.float64)
        mean = float(np.mean(array))
        std = float(np.std(array))
        return cls(mean=mean, scale=std)
