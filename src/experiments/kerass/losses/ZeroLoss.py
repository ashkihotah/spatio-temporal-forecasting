import keras

class ZeroLoss(keras.losses.Loss):
    """
    A loss function that always returns zero, 
    effectively ignoring the loss contribution.
    Useful for scenarios where you want to disable 
    certain loss components during training.
    """
    def __init__(self, name="zero_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        # Return a tensor of zeros with the same shape as the predictions
        return keras.backend.zeros_like(y_pred)