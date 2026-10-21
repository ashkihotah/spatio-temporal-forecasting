import keras
from keras import ops

class NodeHomoscedasticLoss(keras.layers.Layer):
    """
    Homoscedastic Uncertainty Loss (Kendall et al., 2018) 
    for spatio-temporal nodes.
    
    Treats each node N as an independent regression task with its own 
    trainable log-variance parameter eta_n = log(sigma_n^2).
    """
    def __init__(
        self, 
        num_nodes, 
        base_loss="mse", 
        name="node_homoscedastic_loss", 
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_nodes = num_nodes
        self.base_loss = base_loss.lower()

    def build(self, input_shape):
        # Shape (1, 1, N, 1) enables automatic broadcasting over (B, P, N, 1)
        self.log_vars = self.add_weight(
            name="node_log_variances",
            shape=(1, 1, self.num_nodes),
            initializer="zeros",  # Initial sigma_n^2 = 1 (log_var = 0)
            trainable=True,
        )
        super().build(input_shape)

    def call(self, y_true, y_pred):
        if self.base_loss == "mse":
            # Gaussian noise model: 0.5 * exp(-eta) * (y - y_hat)^2 + 0.5 * eta
            elementwise_base = ops.square(y_true - y_pred)
            weighted_loss = 0.5 * ops.exp(-self.log_vars) * elementwise_base + 0.5 * self.log_vars

        elif self.base_loss == "mae":
            # Laplace noise model: exp(-s) * |y - y_hat| + s, where s = log(sigma)
            elementwise_base = ops.abs(y_true - y_pred)
            # s_n = 0.5 * eta_n
            log_std = 0.5 * self.log_vars
            weighted_loss = ops.exp(-log_std) * elementwise_base + log_std

        else:
            raise ValueError(f"Unsupported base_loss: '{self.base_loss}'. Choose 'mse' or 'mae'.")

        # Mean reduction across Batch (B), Horizon (P), Nodes (N), and Features (F=1)
        result = ops.mean(weighted_loss)
        self.add_loss(result)
        return result

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_nodes": self.num_nodes,
            "base_loss": self.base_loss,
        })
        return config