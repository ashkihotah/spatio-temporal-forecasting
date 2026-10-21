import numpy as np
import keras

class KerasScaler(keras.layers.Layer):
    """Abstract Base Class for backend-agnostic Keras 3 scalers."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def normalize(self, x):
        raise NotImplementedError

    def denormalize(self, x):
        raise NotImplementedError

    def call(self, inputs, mode: str = "normalize"):
        """Allows direct layer calling via __call__."""
        if mode == "normalize":
            return self.normalize(inputs)
        elif mode == "denormalize":
            return self.denormalize(inputs)
        raise ValueError(f"Mode must be 'normalize' or 'denormalize', received: {mode}")

    def to_dict(self) -> dict:
        """Hydra/config dictionary representation."""
        return {
            "_target_": f"{self.__class__.__module__}" # .{self.__class__.__name__}
        }

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "KerasScaler":
        raise NotImplementedError(
            "This method should be implemented in subclasses "
            "to create a scaler from dataset statistics."
        )
