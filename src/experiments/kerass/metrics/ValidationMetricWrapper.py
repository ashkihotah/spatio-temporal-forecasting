import keras

class ValidationMetricWrapper(keras.metrics.Metric):
    """
    A generic Keras Metric wrapper that takes any standard Keras regression 
    metric, denormalizes targets and predictions to physical scales, 
    and forwards them to the wrapped metric for evaluation.
    """

    def __init__(
        self,
        metric: keras.metrics.Metric,
        target_scalers: list,
        name: str = None,
        **kwargs,
    ):
        metric_name = name if name is not None else f"raw_{metric.name}"
        super().__init__(name=metric_name, **kwargs)

        self.metric = keras.metrics.deserialize(keras.metrics.serialize(metric))
        self.target_scalers = list(target_scalers)

    def _denormalize(self, normalized_tensor):
        """
        Splits the tensor along the last axis and denormalizes features individually.

        Parameters
        ----------
        normalized_tensor : Tensor
            The normalized tensor of shape (..., F), where F is the number of features.

        Returns
        -------
        Tensor
            The denormalized tensor of the same shape as the input.
        """
        normalized_tensor = keras.ops.convert_to_tensor(normalized_tensor)
        num_features = len(self.target_scalers)
        
        # Split along the last dimension (features) -> 
        # -> Returns list of [..., 1] tensors
        slices = keras.ops.split(normalized_tensor, num_features, axis=-1)

        processed_slices = []
        for idx, slice_tensor in enumerate(slices):
            scaler = self.target_scalers[idx]
            processed_slices.append(scaler.denormalize(slice_tensor))

        return keras.ops.concatenate(processed_slices, axis=-1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 1. Denormalize targets and predictions to the raw scale
        y_true_phys = self._denormalize(y_true)
        y_pred_phys = self._denormalize(y_pred)

        # 2. Update the state of the wrapped metric using raw values
        self.metric.update_state(
            y_true_phys, y_pred_phys, 
            sample_weight=sample_weight
        )

    def result(self):
        # Delegate result computation to the wrapped metric
        return self.metric.result()

    def reset_state(self):
        # Delegate state reset to the wrapped metric
        self.metric.reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({
            "metric": keras.saving.serialize_keras_object(self.metric),
            "target_scalers": [
                keras.saving.serialize_keras_object(s) 
                for s in self.target_scalers
            ],
        })
        return config

    @classmethod
    def from_config(cls, config):
        metric = keras.saving.deserialize_keras_object(config.pop("metric"))
        target_scalers_config = config.pop("target_scalers")
        target_scalers = [
            keras.saving.deserialize_keras_object(s) 
            for s in target_scalers_config
        ]
        return cls(metric=metric, target_scalers=target_scalers, **config)