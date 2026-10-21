import keras

from dvc_mlops.producing import AsyncBufferedStageDumper

class StepAnnealingCallback(keras.callbacks.Callback):
    """Updates the step counter of DynamicLossMaskAnnealing at every epoch or batch."""

    def __init__(self, loss_instance, dumper: AsyncBufferedStageDumper, metric_factory_fn: callable = None):
        super().__init__()
        self.loss_instance = loss_instance
        self.dumper = dumper
        self._metric_factory_fn = metric_factory_fn

    def on_epoch_begin(self, epoch, logs=None):
        if self.loss_instance.mode == "epoch":
            self.loss_instance.current_step.assign(float(epoch + 1))

        self.dumper.dump_metric(
            self._metric_factory_fn(
                name="active_horizon_per_epoch", 
                value=float(self.loss_instance._get_active_horizon()), 
                epoch=int(epoch),
                subset="train"
            )
        )

    def on_train_batch_begin(self, batch, logs=None):
        if self.loss_instance.mode == "batch":
            current_val = float(self.loss_instance.current_step.value)
            self.loss_instance.current_step.assign(current_val + 1.0)

        self.dumper.dump_metric(
            self._metric_factory_fn(
                name="active_horizon_per_batch", 
                value=float(self.loss_instance._get_active_horizon()), 
                epoch=int(self.loss_instance.current_step),
                subset="train"
            )
        )
