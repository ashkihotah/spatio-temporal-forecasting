import keras

@keras.saving.register_keras_serializable()
class SymmetricMeanAbsolutePercentageError(keras.metrics.Mean):
    """Backend-agnostic SMAPE Metric matching the evaluate() formula:
    
    |y_true - y_pred| / (|y_true| + |y_pred|) * 100
    """
    def __init__(self, name="smape", dtype=None):
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = keras.ops.cast(y_true, self.dtype)
        y_pred = keras.ops.cast(y_pred, self.dtype)
        
        abs_diff = keras.ops.abs(y_true - y_pred)
        denominator = keras.ops.abs(y_true) + keras.ops.abs(y_pred)
        
        # Prevent division by zero
        denominator = keras.ops.maximum(denominator, keras.backend.epsilon())
        
        apes = (abs_diff / denominator) * 100.0
        
        # Delegates state updating and tracking to keras.metrics.Mean
        return super().update_state(apes, sample_weight=sample_weight)