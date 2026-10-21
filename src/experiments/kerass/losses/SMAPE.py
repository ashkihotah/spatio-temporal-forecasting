import keras

@keras.saving.register_keras_serializable()
class SymmetricMeanAbsolutePercentageError(keras.losses.Loss):
    """Backend-agnostic SMAPE Loss matching the evaluate() formula:
    
    |y_true - y_pred| / (|y_true| + |y_pred|) * 100
    """
    def __init__(self, name="smape", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        y_pred = keras.ops.cast(y_pred, y_true.dtype)
        
        abs_diff = keras.ops.abs(y_true - y_pred)
        denominator = keras.ops.abs(y_true) + keras.ops.abs(y_pred)
        
        # Prevent division by zero
        denominator = keras.ops.maximum(denominator, keras.backend.epsilon())
        
        apes = (abs_diff / denominator) * 100.0
        return keras.ops.mean(apes, axis=-1)