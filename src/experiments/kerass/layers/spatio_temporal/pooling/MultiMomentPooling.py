import keras
from typing import Tuple

class MultiMomentPooling(keras.layers.Layer):

    def __init__(
        self,
        output_dim: int = None,
        include_skewness: bool = True,
        include_extremes: bool = True,
        include_dirichlet_energy: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.include_skewness = include_skewness
        self.include_extremes = include_extremes
        self.include_dirichlet_energy = include_dirichlet_energy

        if self.output_dim is not None:
            self.projection = keras.layers.Dense(
                self.output_dim, name="moment_projection"
            )

    def build(self, input_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]):
        x_shape, a_shape = input_shape
        feature_dim = x_shape[-1]

        # Calculate concatenated descriptors dimension
        total_descriptors = 2 * feature_dim  # Mean + Std
        if self.include_skewness:
            total_descriptors += feature_dim
        if self.include_extremes:
            total_descriptors += 2 * feature_dim  # Min + Max
        if self.include_dirichlet_energy:
            total_descriptors += 1  # Scalar Dirichlet energy

        if self.output_dim is not None:
            # Build downstream projection layer
            self.projection.build((None, total_descriptors))
        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
        training: bool = False,
    ) -> keras.KerasTensor:
        x, a = inputs  # x: (B, N, F), a: (B, N, N)
        dtype = keras.ops.dtype(x)
        eps = 1e-7

        n_nodes = keras.ops.cast(keras.ops.shape(x)[1], dtype=dtype)
        f_dim = keras.ops.cast(keras.ops.shape(x)[2], dtype=dtype)

        descriptors = []

        # 1. First Moment (Mean)
        mean = keras.ops.mean(x, axis=1)  # (B, F)
        descriptors.append(mean)

        # 2. Second Central Moment (Standard Deviation)
        diff = x - keras.ops.expand_dims(mean, axis=1)
        variance = keras.ops.mean(keras.ops.square(diff), axis=1)  # (B, F)
        std = keras.ops.sqrt(variance + eps)
        descriptors.append(std)

        # 3. Third Standardized Moment (Skewness)
        if self.include_skewness:
            standardized_diff = diff / (
                keras.ops.expand_dims(std, axis=1) + eps
            )
            skewness = keras.ops.mean(
                keras.ops.power(standardized_diff, 3.0), axis=1
            )
            descriptors.append(skewness)

        # 4. Extremes (Min & Max Envelopes)
        if self.include_extremes:
            min_val = keras.ops.min(x, axis=1)
            max_val = keras.ops.max(x, axis=1)
            descriptors.extend([min_val, max_val])

        # 5. Dirichlet Energy on Adjacency
        if self.include_dirichlet_energy:
            # Degree term: sum_i d_i * ||x_i||^2
            degrees = keras.ops.sum(a, axis=-1, keepdims=True)  # (B, N, 1)
            deg_term = keras.ops.sum(
                degrees * keras.ops.square(x), axis=(1, 2), keepdims=True
            )  # (B, 1)

            # Adjacency term: sum_{i,j} A_{ij} * (x_i . x_j)
            ax = keras.ops.matmul(a, x)  # (B, N, F)
            adj_term = keras.ops.sum(
                x * ax, axis=(1, 2), keepdims=True
            )  # (B, 1)

            dirichlet_energy = (deg_term - adj_term) / (2.0 * n_nodes * f_dim + eps)
            descriptors.append(dirichlet_energy)

        # Concatenate into fixed-size summary vector: (B, total_descriptors)
        summary = keras.ops.concatenate(descriptors, axis=-1)

        # Output projection: (B, D)
        if self.output_dim is not None:
            summary = self.projection(summary)

        return summary

    def compute_output_shape(self, input_shape):
        x_shape, _ = input_shape
        batch_size = x_shape[0]

        total_descriptors = 2 * x_shape[-1]  # Mean + Std
        if self.include_skewness:
            total_descriptors += x_shape[-1]
        if self.include_extremes:
            total_descriptors += 2 * x_shape[-1]  # Min + Max
        if self.include_dirichlet_energy:
            total_descriptors += 1  # Scalar Dirichlet energy

        if self.output_dim is not None:
            return (batch_size, self.output_dim)
        else:
            return (batch_size, total_descriptors)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "include_skewness": self.include_skewness,
                "include_extremes": self.include_extremes,
                "include_dirichlet_energy": self.include_dirichlet_energy,
            }
        )
        return config