import keras

from .KerasScaler import KerasScaler

class ScalingLayer(keras.layers.Layer):
    """
    Applies feature-wise scaling using a 
    list of custom `KerasScaler` instances.
    """

    def __init__(
        self, 
        scalers: list[KerasScaler], 
        mode: str = "normalize", 
        **kwargs
    ):
        super().__init__(**kwargs)
        if mode not in ["normalize", "denormalize"]:
            raise ValueError("Mode must be 'normalize' or 'denormalize'")
        self.scalers = list(scalers)
        self.mode = mode

    def call(self, inputs):
        inputs = keras.ops.convert_to_tensor(inputs)
        num_features = len(self.scalers)
        
        # Split along the last axis
        slices = keras.ops.split(inputs, num_features, axis=-1)

        processed_slices = []
        for idx, slice_tensor in enumerate(slices):
            scaler = self.scalers[idx]
            method = getattr(scaler, self.mode)
            processed_slices.append(method(slice_tensor))

        return keras.ops.concatenate(processed_slices, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "scalers": [
                keras.saving.serialize_keras_object(s) 
                for s in self.scalers
            ],
            "mode": self.mode,
        })
        return config

    @classmethod
    def from_config(cls, config):
        scalers_config = config.pop("scalers")
        scalers = [
            keras.saving.deserialize_keras_object(s) 
            for s in scalers_config
        ]
        return cls(scalers=scalers, **config)