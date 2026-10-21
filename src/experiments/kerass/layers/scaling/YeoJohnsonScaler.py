import numpy as np
import keras

from .KerasScaler import KerasScaler

class YeoJohnsonScaler(KerasScaler):
    """Backend-agnostic mathematical implementation of Yeo-Johnson Power Transform."""

    def __init__(self, lmbda: float = 1.0, mean: float = 0.0, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._init_lmbda = float(lmbda)
        self._init_mean = float(mean)
        self._init_scale = float(scale)
        self.lmbda = self.add_weight(
            name="lmbda",
            shape=(),
            initializer=keras.initializers.Constant(self._init_lmbda),
            trainable=False,
            dtype="float32",
        )
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

    def _transform(self, x):
        lmbda = keras.ops.cast(self.lmbda, x.dtype)
        two = keras.ops.cast(2.0, x.dtype)
        pos_mask = x >= 0.0

        pos_log = keras.ops.log1p(x)
        pos_scaled = keras.ops.expm1(lmbda * pos_log)
        pos_general = keras.ops.divide_no_nan(pos_scaled, lmbda)
        pos_val = keras.ops.where(keras.ops.equal(lmbda, 0.0), pos_log, pos_general)

        neg_log = keras.ops.log1p(-x)
        neg_lambda = two - lmbda
        neg_scaled = keras.ops.expm1(neg_lambda * neg_log)
        neg_general = -keras.ops.divide_no_nan(neg_scaled, neg_lambda)
        neg_val = keras.ops.where(keras.ops.equal(lmbda, 2.0), -neg_log, neg_general)

        return keras.ops.where(pos_mask, pos_val, neg_val)

    def _inverse_transform(self, x):
        lmbda = keras.ops.cast(self.lmbda, x.dtype)
        two = keras.ops.cast(2.0, x.dtype)
        pos_mask = x >= 0.0

        pos_scaled = keras.ops.divide_no_nan(keras.ops.log1p(x * lmbda), lmbda)
        pos_general = keras.ops.expm1(pos_scaled)
        pos_val = keras.ops.where(keras.ops.equal(lmbda, 0.0), keras.ops.expm1(x), pos_general)

        neg_lambda = two - lmbda
        neg_scaled = keras.ops.divide_no_nan(keras.ops.log1p(-x * neg_lambda), neg_lambda)
        neg_general = -keras.ops.expm1(neg_scaled)
        neg_val = keras.ops.where(keras.ops.equal(lmbda, 2.0), -keras.ops.expm1(-x), neg_general)

        return keras.ops.where(pos_mask, pos_val, neg_val)

    def normalize(self, x):
        x = keras.ops.convert_to_tensor(x)
        yj_transformed = self._transform(x)
        mean = keras.ops.cast(self.mean, x.dtype)
        scale = keras.ops.cast(self.scale, x.dtype)
        return keras.ops.divide_no_nan(yj_transformed - mean, scale)

    def denormalize(self, x):
        x = keras.ops.convert_to_tensor(x)
        mean = keras.ops.cast(self.mean, x.dtype)
        scale = keras.ops.cast(self.scale, x.dtype)
        unscaled = (x * scale) + mean
        return self._inverse_transform(unscaled)

    def get_config(self):
        config = super().get_config()
        config.update({
            "lmbda": float(keras.ops.convert_to_numpy(self.lmbda)),
            "mean": float(keras.ops.convert_to_numpy(self.mean)),
            "scale": float(keras.ops.convert_to_numpy(self.scale)),
        })
        return config

    def to_dict(self) -> dict:
        self_dict = super().to_dict()
        self_dict.update({
            "lmbda": float(keras.ops.convert_to_numpy(self.lmbda)),
            "mean": float(keras.ops.convert_to_numpy(self.mean)),
            "scale": float(keras.ops.convert_to_numpy(self.scale)),
        })
        return self_dict

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "YeoJohnsonScaler":
        lmbda = 1.0  # Replace with MLE lambda estimation if needed
        array = np.asarray(array, dtype=np.float64)
        mean = float(np.mean(array))
        std = float(np.std(array))
        return cls(lmbda=lmbda, mean=mean, scale=std)
