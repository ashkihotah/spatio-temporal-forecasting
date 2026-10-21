import numpy as np
import keras

class DynamicLossMaskAnnealing(keras.losses.Loss):
    """Production-ready dynamic loss-mask annealing for time-series / seq2seq models in Keras 3.

    Correctly guarantees equal distribution across all integer horizons from min_p to max_p,
    ensuring max_p receives full training budget at the end of schedule.
    """

    def __init__(
        self,
        total_steps: int,
        mode: str, # "batch" or "epoch"
        min_p: int,
        max_p: int,
        base_loss="mae",
        schedule="linear",
        name="dynamic_loss_mask_annealing",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.base_loss = keras.losses.get(base_loss)
        self.total_steps = float(total_steps)
        self.schedule = schedule.lower()
        self.min_p = float(min_p)
        self.max_p = float(max_p)
        self.mode = mode.lower()

        # Internal progress tracker (starts at step 1.0)
        self.current_step = keras.Variable(
            initializer=1.0,
            trainable=False,
            dtype="float32",
            name="current_progress_step",
        )

    def _get_active_horizon(self):
        """Calculates active horizon P_active using 1-based progress scaling so max_p is fully utilized."""
        # Normalize step k in [1, total_steps] to ratio in [0.0, 1.0]
        # Prevents division by zero if total_steps == 1
        denom = keras.ops.maximum(self.total_steps - 1.0, 1.0)
        progress_ratio = keras.ops.clip(
            (self.current_step - 1.0) / denom, 0.0, 1.0
        )

        if self.schedule == "linear":
            mapped_ratio = progress_ratio

        elif self.schedule == "cosine":
            # Smooth S-curve transition
            mapped_ratio = 0.5 * (1.0 - keras.ops.cos(progress_ratio * np.pi))

        elif self.schedule == "exponential":
            # Slow start at min_p, fast acceleration towards max_p
            mapped_ratio = (keras.ops.exp(progress_ratio * 3.0) - 1.0) / (
                np.exp(3.0) - 1.0
            )

        elif self.schedule == "step":
            # Discrete step jumps across total stages
            num_stages = 4.0
            mapped_ratio = (
                keras.ops.floor(progress_ratio * num_stages) / (num_stages - 1.0)
            )
            mapped_ratio = keras.ops.clip(mapped_ratio, 0.0, 1.0)

        else:
            raise ValueError(
                f"Unsupported schedule: '{self.schedule}'. Choose from "
                "'linear', 'cosine', 'exponential', 'step'."
            )

        # Continuous interpolation between min_p and max_p followed by round()
        p_float = self.min_p + mapped_ratio * (self.max_p - self.min_p)
        p_active = keras.ops.round(keras.ops.clip(p_float, self.min_p, self.max_p))

        return keras.ops.cast(p_active, "int32")

    def call(self, y_true, y_pred):
        """Computes masked loss across batch B and horizon P."""
        loss_tensor = self.base_loss(y_true, y_pred)

        p_active = self._get_active_horizon()
        seq_length = keras.ops.shape(y_pred)[1]  # Horizon P (e.g., 19)

        step_indices = keras.ops.arange(0, seq_length, dtype="int32")
        mask = keras.ops.cast(step_indices < p_active, "float32")

        # Reshape mask to broadcast across (B, P, N, 1)
        for _ in range(len(keras.ops.shape(loss_tensor)) - 2):
            mask = keras.ops.expand_dims(mask, axis=-1)
        mask = keras.ops.expand_dims(mask, axis=0)

        masked_loss = loss_tensor * mask

        batch_size = keras.ops.cast(keras.ops.shape(loss_tensor)[0], "float32")
        active_elements = keras.ops.sum(mask) * batch_size

        return keras.ops.sum(masked_loss) / (active_elements + 1e-7)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "base_loss": keras.losses.serialize(self.base_loss),
                "total_steps": self.total_steps,
                "schedule": self.schedule,
                "min_p": self.min_p,
                "max_p": self.max_p,
                "mode": self.mode,
            }
        )
        return config
