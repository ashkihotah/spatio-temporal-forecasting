import keras

# @keras.saving.register_keras_serializable()
# class TrainableLocalPoolingFilter(keras.layers.Layer):
#     """
#     Computes a dynamically normalized Kipf & Welling GCN adjacency matrix
#     with a trainable self-loop parameter alpha:
    
#         A_norm = D^(-1/2) * (A + softplus(alpha) * I) * D^(-1/2)
#     """
#     def __init__(self, initial_alpha: float = 1.0, **kwargs):
#         super().__init__(**kwargs)
#         self.initial_alpha = initial_alpha

#     def build(self, input_shape):
#         # Learnable self-loop scale factor (unconstrained initial weight)
#         self.alpha_raw = self.add_weight(
#             name="alpha_raw",
#             shape=(),
#             initializer=keras.initializers.Constant(self.initial_alpha),
#             trainable=True,
#         )
#         super().build(input_shape)

#     def call(self, A: keras.KerasTensor) -> keras.KerasTensor:
#         # 1. Enforce alpha > 0 dynamically
#         alpha = keras.ops.softplus(self.alpha_raw)
        
#         # 2. Add weighted self-loops: A_tilde = A + alpha * I
#         n_nodes = keras.ops.shape(A)[-1]
#         I = keras.ops.eye(n_nodes, dtype=A.dtype)
#         A_tilde = A + alpha * I
        
#         # 3. Compute row sums (degrees)
#         degrees = keras.ops.sum(A_tilde, axis=-1)
        
#         # 4. Compute D^(-1/2) safely
#         deg_inv_sqrt = keras.ops.power(degrees, -0.5)
        
#         # Replace infinities/NaNs resulting from zero-degree nodes with 0.0
#         deg_inv_sqrt = keras.ops.nan_to_num(
#             deg_inv_sqrt, 
#             nan=0.0, 
#             posinf=0.0, 
#             neginf=0.0
#         )
        
#         # 5. Symmetric Normalization via broadcasting: D^(-1/2)_i * A_ij * D^(-1/2)_j
#         D_i = keras.ops.expand_dims(deg_inv_sqrt, axis=-1)  # Shape: (..., N, 1)
#         D_j = keras.ops.expand_dims(deg_inv_sqrt, axis=-2)  # Shape: (..., 1, N)
        
#         A_norm = A_tilde * D_i * D_j
#         return A_norm

#     def get_config(self):
#         config = super().get_config()
#         config.update({"initial_alpha": self.initial_alpha})
#         return config

class TrainableLocalPoolingFilter(keras.layers.Layer):
    """
    Computes normalized A once per batch/pass.
    Must be called OUTSIDE the time-unrolling loop (e.g. before the LSTM layer).
    """
    def __init__(self, initial_alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_alpha = initial_alpha

    def build(self, input_shape):
        self.alpha_raw = self.add_weight(
            name="alpha_raw",
            shape=(),
            initializer=keras.initializers.Constant(self.initial_alpha),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, A: keras.KerasTensor) -> keras.KerasTensor:
        # Enforce positive alpha via softplus + epsilon to completely eliminate NaNs/Infs!
        alpha = keras.ops.softplus(self.alpha_raw) + 1e-7
        
        n_nodes = keras.ops.shape(A)[-1]
        I = keras.ops.eye(n_nodes, dtype=A.dtype)
        
        # A_tilde = A + alpha * I
        A_tilde = A + alpha * I
        
        # Degrees are strictly positive because alpha > 1e-7 and I has diagonal 1.0
        # This completely removes the need for nan_to_num!
        degrees = keras.ops.sum(A_tilde, axis=-1)
        deg_inv_sqrt = keras.ops.rsqrt(degrees)  # Fast GPU reciprocal square root!
        
        # Symmetric Normalization via broadcasting
        D_i = keras.ops.expand_dims(deg_inv_sqrt, axis=-1)
        D_j = keras.ops.expand_dims(deg_inv_sqrt, axis=-2)
        
        return A_tilde * D_i * D_j

    def get_config(self):
        config = super().get_config()
        config.update({"initial_alpha": self.initial_alpha})
        return config