import keras

class AxisFlexibleDense(keras.layers.Layer):
    """
    A dense projection layer that allows weight sharing or independent weights
    across specified intermediate axes.
    """

    def __init__(
        self,
        units: int = None,
        axes: tuple = (),
        use_bias: bool = True,
        kernel_params: dict = None,
        bias_params: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units

        # All axes specified in `axes` are the ones that 
        # will have independent weights. All other 
        # intermediate axes will share weights.

        # Gracefully normalize axes input
        if axes is None:
            self.axes = ()
        elif isinstance(axes, (list, tuple)):
            self.axes = tuple(axes)
        else:
            self.axes = (axes,)

        self.use_bias = use_bias
        if kernel_params is not None:
            self.kernel_params = dict(kernel_params)
        else:
            self.kernel_params = {}

        if bias_params is not None:
            self.bias_params = dict(bias_params)
        else:
            self.bias_params = {}

        self.einsum_equation = None

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        rank = len(input_shape)

        if self.units is None:
            self.units = input_shape[-1]  # Default to input feature size

        if rank <= 2:
            raise ValueError(
                f"AxisFlexibleDense expects input rank > 2, received rank {rank}."
            )

        num_intermediate = rank - 2

        # 1. Normalize user-specified axes to absolute indices
        normalized_axes = []
        for axis in self.axes:
            if axis < 0:
                axis = rank + axis
            if axis <= 0 or axis >= rank - 1:
                raise ValueError(
                    f"Invalid axis {axis}. Intermediate axes must be in range [1, {rank - 2}]."
                )
            normalized_axes.append(axis)

        # 2. Build kernel shape & bias shape dynamically
        shared_mask = []
        kernel_shape = []
        bias_shape = [1]  # Leading 1 for batch dimension broadcasting

        for i in range(1, rank - 1):
            dim_size = input_shape[i]
            if dim_size is None:
                raise ValueError(
                    f"Intermediate dimension at axis {i} is dynamic (None). "
                    f"To build non-shared weights, all intermediate dimensions must be static."
                )

            if i in normalized_axes:
                shared_mask.append(False)
                kernel_shape.append(dim_size)
                bias_shape.append(dim_size)
            else:
                shared_mask.append(True)
                # Shared dimensions are size 1 in the bias for broadcasting
                bias_shape.append(1)

        input_features = input_shape[-1]
        if input_features is None:
            raise ValueError("Feature dimension (last axis) must be static.")

        kernel_shape.extend([input_features, self.units])
        bias_shape.append(self.units)

        # 3. Add Weight Kernel with default initializer if not provided
        kernel_kwargs = {"initializer": "glorot_uniform"}
        kernel_kwargs.update(self.kernel_params)

        self.kernel = self.add_weight(
            name="kernel",
            shape=tuple(kernel_shape),
            trainable=True,
            **kernel_kwargs,
        )

        # 4. Add Bias Vector with default initializer if not provided
        if self.use_bias:
            bias_kwargs = {"initializer": "zeros"}
            bias_kwargs.update(self.bias_params)

            self.bias = self.add_weight(
                name="bias",
                shape=tuple(bias_shape),
                trainable=True,
                **bias_kwargs,
            )
        else:
            self.bias = None

        # 5. Build the Dynamic Einsum Equation
        char_pool = "cdeghijklmnopqrstuvw"
        if num_intermediate > len(char_pool):
            raise ValueError(f"Too many intermediate dimensions (max allowed: {len(char_pool)})")

        input_chars = [char_pool[idx] for idx in range(num_intermediate)]

        # 'b' + intermediate_chars + 'f' (input features)
        input_pattern = "b" + "".join(input_chars) + "f"

        # Keep only the characters corresponding to the non-shared axes
        kernel_chars = [
            input_chars[idx] 
            for idx, is_shared in enumerate(shared_mask) 
            if not is_shared
        ]
        kernel_pattern = "".join(kernel_chars) + "fz"  
        # 'z' represents output units

        # 'b' + intermediate_chars + 'z'
        output_pattern = "b" + "".join(input_chars) + "z"

        self.einsum_equation = f"{input_pattern},{kernel_pattern}->{output_pattern}"

        super().build(input_shape)

    def call(self, inputs):
        # 1. Compute projection contraction via backend-agnostic einsum
        projected = keras.ops.einsum(self.einsum_equation, inputs, self.kernel)

        # 2. Add bias (broadcasts automatically across shared dimensions)
        if self.bias is not None:
            projected = projected + self.bias

        return projected

    def compute_output_shape(self, input_shape):
        """Idiomatic Keras 3 symbolic shape propagation."""
        input_shape = tuple(input_shape)
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "axes": self.axes,
            "use_bias": self.use_bias,
            "kernel_params": self.kernel_params,
            "bias_params": self.bias_params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)