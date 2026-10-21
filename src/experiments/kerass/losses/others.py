from keras import ops
from keras.losses import Loss
import numpy as np

class SpatioTemporalPearsonLoss(Loss):
    """
    Computes (1 - r) where r is the Pearson Correlation Coefficient 
    along the temporal horizon axis 'P' (Axis 1).
    """
    def __init__(self, name: str = "spatio_temporal_pearson_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        # Shapes: B x P x N x F
        # We compute mean and variance along Axis 1 (the temporal dimension P)
        axis = 1
        
        # 1. Compute means
        mean_true = ops.mean(y_true, axis=axis, keepdims=True)
        mean_pred = ops.mean(y_pred, axis=axis, keepdims=True)

        # 2. Compute deviances from the mean
        true_dev = y_true - mean_true
        pred_dev = y_pred - mean_pred

        # 3. Compute Covariance: sum( (y - mean_y) * (y_hat - mean_y_hat) )
        covariance = ops.sum(true_dev * pred_dev, axis=axis)

        # 4. Compute Standard Deviations
        var_true = ops.sum(ops.square(true_dev), axis=axis)
        var_pred = ops.sum(ops.square(pred_dev), axis=axis)
        
        # Add epsilon to prevent division-by-zero during dark/zero-variance hours
        std_product = ops.sqrt(var_true * var_pred) + 1e-8

        # 5. Compute Pearson r
        r = covariance / std_product

        # Return (1 - r) as a minimizable loss
        return ops.mean(1.0 - r)

class SpatioTemporalCompositeLoss(Loss):
    """
    Combines Capacity-Weighted MSE (for magnitude) with Pearson Loss (for shape)
    to enforce physically consistent, well-shaped solar trajectories.
    """
    def __init__(
        self,
        lambda_coef: float = 0.5,
        name: str = "spatio_temporal_composite_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.lambda_coef = float(lambda_coef)
        self.pearson_loss_fn = SpatioTemporalPearsonLoss()

    def call(self, y_true, y_pred):
        diff = ops.abs(y_true - y_pred) 
        mae_loss = ops.mean(diff)

        # Part B: Pearson Correlation Loss (Shape/Phase Specialist)
        pearson_loss = self.pearson_loss_fn(y_true, y_pred)

        # Weighted combination
        return mae_loss + (self.lambda_coef * pearson_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "capacities": self.capacities.numpy().tolist(),
            "lambda_coef": self.lambda_coef,
        })
        return config

class SpatioTemporalKGELoss(Loss):
    """
    Computes the Kling-Gupta Efficiency (KGE) Loss over the temporal horizon dimension 'P' (Axis 1).
    Optimized for Spatio-Temporal B x P x N x F tensor shapes.
    """
    def __init__(self, epsilon: float = 1e-8, name: str = "spatio_temporal_kge_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.epsilon = float(epsilon)

    def call(self, y_true, y_pred):
        # Target shape: B x P x N x F
        # Correlation, Bias, and Standard Deviation are calculated along Axis 1 (P)
        axis = 1

        # 1. Compute Means: Shape (B, 1, N, F)
        mean_true = ops.mean(y_true, axis=axis, keepdims=True)
        mean_pred = ops.mean(y_pred, axis=axis, keepdims=True)

        # 2. Compute Variances and Standard Deviations: Shape (B, 1, N, F)
        var_true = ops.mean(ops.square(y_true - mean_true), axis=axis, keepdims=True)
        var_pred = ops.mean(ops.square(y_pred - mean_pred), axis=axis, keepdims=True)
        
        std_true = ops.sqrt(var_true + self.epsilon)
        std_pred = ops.sqrt(var_pred + self.epsilon)

        # 3. Compute Covariance and Pearson Correlation (r): Shape (B, 1, N, F)
        covariance = ops.mean((y_true - mean_true) * (y_pred - mean_pred), axis=axis, keepdims=True)
        r = covariance / (std_true * std_pred + self.epsilon)

        # 4. Compute Bias Ratio (Beta): Shape (B, 1, N, F)
        beta = mean_pred / (mean_true + self.epsilon)

        # 5. Compute Variability Ratio (Alpha): Shape (B, 1, N, F)
        alpha = std_pred / (std_true + self.epsilon)

        # 6. Compute KGE and Loss
        # Perfect predictions have r=1, beta=1, alpha=1, resulting in KGE = 1.0
        kge = 1.0 - ops.sqrt(
            ops.square(r - 1.0) + 
            ops.square(beta - 1.0) + 
            ops.square(alpha - 1.0)
        )

        # Minimize (1.0 - KGE)
        loss = 1.0 - kge
        return ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

class CapacityWeightedGraphLaplacianLoss(Loss):
    """
    Computes the Capacity-Weighted Masked MSE with Graph Laplacian Regularization.
    
    This loss normalizes errors by each node's peak capacity, penalizes spurious
    night-time oscillations via a mask, and enforces geographic prediction consistency
    using the spatial Graph Laplacian matrix.
    """
    def __init__(
        self,
        capacities: list,
        laplacian_matrix: np.ndarray,
        night_weight: float = 5.0,
        spatial_coefficient: float = 0.1,
        name: str = "capacity_weighted_graph_laplacian_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        # Convert inputs to backend-agnostic tensors
        self.capacities = ops.convert_to_tensor(capacities, dtype="float32")
        self.laplacian_matrix = ops.convert_to_tensor(laplacian_matrix, dtype="float32")
        self.night_weight = float(night_weight)
        self.spatial_coefficient = float(spatial_coefficient)

        # Reshape capacities to (1, num_nodes, 1) for clean broadcasting
        self.capacities_reshaped = ops.expand_dims(ops.expand_dims(self.capacities, axis=0), axis=0)
        self.laplacian_matrix = ops.expand_dims(laplacian_matrix, axis=0)

    def call(self, y_true, y_pred):
        # 1. Dynamically compute the night/day mask where y_true is exactly 0.0
        # If the sun is down (y_true == 0), we apply the heavier night_weight penalty
        mask = ops.where(ops.equal(y_true, 0.0), self.night_weight, 1.0)

        broadcasted_capacities = ops.broadcast_to(self.capacities_reshaped, ops.shape(y_true))
        broadcasted_laplacian = ops.broadcast_to(self.laplacian_matrix, ops.shape(y_true)[:2] + ops.shape(self.laplacian_matrix)[1:])

        # 2. Capacity Normalization: Scale absolute errors by the plant's peak capacity
        normalized_diff = (y_true - y_pred) / (broadcasted_capacities + 1e-8)
        weighted_squared_error = mask * ops.square(normalized_diff)
        reconstruction_loss = ops.mean(weighted_squared_error)

        # 3. Spatial Regularization: Compute the quadratic form of the Graph Laplacian
        # y_pred shape: (batch_size, num_nodes, horizon)
        # We use einsum to multiply the Laplacian L (N, N) with prediction Y (B, N, H)
        # L_y represents L multiplied by Y for each batch and time-step
        L_y = ops.einsum("ij,bjk->bik", broadcasted_laplacian, y_pred)
        
        # Calculate the quadratic form: Y_transpose * L * Y
        spatial_variance = ops.sum(y_pred * L_y, axis=1) # Shape: (batch_size, horizon)
        spatial_loss = ops.mean(spatial_variance)

        # Combine losses
        return reconstruction_loss + (self.spatial_coefficient * spatial_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "capacities": self.capacities.numpy().tolist(),
            "laplacian_matrix": self.laplacian_matrix.numpy().tolist(),
            "night_weight": self.night_weight,
            "spatial_coefficient": self.spatial_coefficient,
        })
        return config

class MAE(Loss):
    def __init__(self, name: str = "mae", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        diff = ops.abs(y_true - y_pred) 
        mae_loss = ops.mean(diff)
        return mae_loss

class SoftDTWLoss(Loss):
    """
    Computes a numerically stable, parallelized, and fully differentiable 
    Soft-Dynamic Time Warping (Soft-DTW) sequence loss.
    
    This version avoids in-place slice updates, making it 100% compatible
    with TensorFlow's GradientTape, PyTorch, and JAX backends.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        name: str = "soft_dtw_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)

    def _softmin(self, a, b, c):
        """Numerically stable soft-minimum approximation using Log-Sum-Exp trick."""
        # Stack inputs along the last axis to shape: (total_sequences, 1, 3)
        inputs = ops.stack([-a / self.gamma, -b / self.gamma, -c / self.gamma], axis=-1)
        max_val = ops.max(inputs, axis=-1, keepdims=True)
        sum_exp = ops.sum(ops.exp(inputs - max_val), axis=-1, keepdims=True)
        res = max_val + ops.log(sum_exp + 1e-10)
        return -self.gamma * ops.squeeze(res, axis=-1)

    def call(self, y_true, y_pred):
        # Shape of y_true & y_pred: (batch_size, num_nodes, horizon)
        batch_size = ops.shape(y_true)[0]
        num_nodes = ops.shape(y_true)[1]
        horizon = ops.shape(y_true)[2]

        # Flatten the spatial dimension into the batch dimension to calculate in parallel
        y_true_flat = ops.reshape(y_true, (-1, horizon))  # Shape: (B * N, H)
        y_pred_flat = ops.reshape(y_pred, (-1, horizon))  # Shape: (B * N, H)
        total_sequences = batch_size * num_nodes

        # Compute pairwise squared distance matrix (D)
        # Shape of D: (total_sequences, horizon, horizon)
        x_expanded = ops.expand_dims(y_true_flat, axis=-1)
        y_expanded = ops.expand_dims(y_pred_flat, axis=-2)
        D = ops.square(x_expanded - y_expanded)

        large_val = 1e8  # Finite proxy for infinity

        # 1. Initialize Row 0
        # R[0, 0] = 0, and R[0, j] = infinity for j > 0
        row_0_list = [ops.zeros((total_sequences, 1))] + [
            ops.full((total_sequences, 1), large_val) for _ in range(horizon)
        ]
        prev_row = ops.concatenate(row_0_list, axis=1)  # Shape: (total_sequences, horizon + 1)

        # 2. Compute DP Table row-by-row to remain fully differentiable
        for i in range(1, horizon + 1):
            # Col 0 is always initialized to large_val (base case)
            curr_row_list = [ops.full((total_sequences, 1), large_val)]
            
            for j in range(1, horizon + 1):
                # Retrieve preceding path values
                r_left = curr_row_list[j - 1]
                r_up = prev_row[:, j : j + 1]
                r_diag = prev_row[:, j - 1 : j]

                # Compute soft-minimum and add local distance cost
                smin = self._softmin(r_left, r_up, r_diag)
                cost = D[:, i - 1, j - 1 : j] + smin
                curr_row_list.append(cost)
            
            # Concatenate list to form the completed row
            prev_row = ops.concatenate(curr_row_list, axis=1)

        # The final Soft-DTW loss value is located at the last cell of the last row
        loss = prev_row[:, horizon]
        return ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma})
        return config