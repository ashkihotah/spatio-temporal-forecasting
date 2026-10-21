import numpy as np
import keras

from .KerasScaler import KerasScaler
from .IdentityScaler import IdentityScaler
from .StandardScaler import StandardScaler
from .Log1pScaler import Log1pScaler
from .MinMaxScaler import MinMaxScaler
from .YeoJohnsonScaler import YeoJohnsonScaler

CLASS_REGISTRY = {
    "IdentityScaler": IdentityScaler,
    "StandardScaler": StandardScaler,
    "Log1pScaler": Log1pScaler,
    "MinMaxScaler": MinMaxScaler,
    "YeoJohnsonScaler": YeoJohnsonScaler,
}

class SequentialScaler(KerasScaler):
    """Sequentially applies a list of KerasScaler transformations."""

    def __init__(self, scalers: list[KerasScaler], **kwargs):
        super().__init__(**kwargs)
        self.scalers = list(scalers)

    def normalize(self, x):
        x = keras.ops.convert_to_tensor(x)
        for scaler in self.scalers:
            x = scaler.normalize(x)
        return x

    def denormalize(self, x):
        x = keras.ops.convert_to_tensor(x)
        for scaler in reversed(self.scalers):
            x = scaler.denormalize(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "scalers": [keras.saving.serialize_keras_object(s) for s in self.scalers]
        })
        return config

    @classmethod
    def from_config(cls, config):
        scalers_config = config.pop("scalers")
        scalers = [keras.saving.deserialize_keras_object(s) for s in scalers_config]
        return cls(scalers=scalers, **config)

    def to_dict(self) -> dict:
        self_dict = super().to_dict()
        self_dict.update({
            "scalers": [scaler.to_dict() for scaler in self.scalers]
        })
        return self_dict

    @classmethod
    def from_numpy(cls, array: np.ndarray, scalers_classes: list[str]) -> "SequentialScaler":
        array = np.asarray(array, dtype=np.float64)
        scalers = []
        for scaler_name in scalers_classes:
            scaler_cls = CLASS_REGISTRY[scaler_name]
            scaler = scaler_cls.from_numpy(array)
            array = keras.ops.convert_to_numpy(scaler.normalize(keras.ops.convert_to_tensor(array)))
            scalers.append(scaler)
        return cls(scalers=scalers)
