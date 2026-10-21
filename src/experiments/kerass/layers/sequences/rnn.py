import optree
import keras

from experiments.kerass.utils import is_node_a_tensor_shape

from .constants import LOGS_KEY, OUTPUTS_KEY, STATE_KEY
from .utils import strip_time_dim

class RNN(keras.layers.Layer):

    def __init__(
        self, 
        cell: keras.layers.Layer, 
        return_outputs: bool = False, 
        return_logs: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cell = cell
        self.return_outputs = return_outputs
        self.return_logs = return_logs

    def build(self, input_shape):
        cell_input_shape = keras.tree.map_shape_structure(
            strip_time_dim, input_shape
        )

        if not self.cell.built:
            self.cell.build(cell_input_shape)

        super().build(input_shape)

    def __get_dynamic_fns(self, inputs, timesteps, training=None):

        def cond(args):
            return args["t"] < timesteps

        def body(args):
            t = args["t"]
            current_state = args[STATE_KEY]

            inputs_at_t = keras.tree.map_structure(
                lambda x: keras.ops.take(x, t, axis=1), inputs
            )
            outputs, next_state, logs, step_loss = self.cell(
                inputs_at_t, current_state,
                training=training
            )

            next_args = {
                "t": t + 1,
                STATE_KEY: next_state,
                "aux_loss": args["aux_loss"] + step_loss,
            }

            if self.return_outputs:
                outputs_buffers = args["outputs_buffers"]
                indices = keras.ops.expand_dims(
                    keras.ops.expand_dims(t, 0), 0
                )
                updated_outputs_buffers = keras.tree.map_structure(
                    lambda buffer, val: keras.ops.scatter_update(
                        buffer, 
                        indices, 
                        keras.ops.expand_dims(val, 0)
                    )  if keras.ops.shape(val)[-1] > 0 else buffer,
                    outputs_buffers, outputs
                )
                next_args["outputs_buffers"] = updated_outputs_buffers

            if self.return_logs:
                logs_buffers = args["logs_buffers"]
                indices = keras.ops.expand_dims(
                    keras.ops.expand_dims(t, 0), 0
                )
                updated_logs_buffers = keras.tree.map_structure(
                    lambda buffer, val: keras.ops.scatter_update(
                        buffer,
                        indices,
                        keras.ops.expand_dims(val, 0)
                    ),
                    logs_buffers, logs
                )
                next_args["logs_buffers"] = updated_logs_buffers

            return next_args

        return cond, body

    def __call_dynamic(
        self, 
        inputs, 
        initial_state,
        timesteps: int, 
        outputs_buffers=None,
        logs_buffers=None,
        training=None
    ):
        cond_fn, body_fn = self.__get_dynamic_fns(
            inputs=inputs,
            timesteps=timesteps,
            training=training
        )

        loop_vars = {
            "t": keras.ops.cast(0, dtype="int32"),
            STATE_KEY: initial_state,
            "aux_loss": keras.ops.zeros((), dtype=keras.ops.dtype(
                keras.tree.flatten(inputs)[0]
            )),
        }

        if self.return_outputs:
            loop_vars["outputs_buffers"] = outputs_buffers

        if self.return_logs:
            loop_vars["logs_buffers"] = logs_buffers

        output = keras.ops.while_loop(
            cond_fn, body_fn, loop_vars, 
            maximum_iterations=timesteps
        )
        
        output.pop("t", None)
        aux_loss = output.pop("aux_loss")
        norm_aux_loss = aux_loss / keras.ops.cast(
            timesteps, dtype=keras.ops.dtype(aux_loss)
        )
        self.add_loss(norm_aux_loss)
        return output

    def call(self, inputs, initial_state = None, training=None):
        flat_inputs = keras.tree.flatten(inputs)

        first_input = flat_inputs[0]
        batch_size = keras.ops.shape(first_input)[0]
        timesteps = keras.ops.shape(first_input)[1]
        dtype = keras.ops.dtype(first_input)

        if initial_state is None:
            initial_state = self.cell.get_initial_state(
                batch_size=batch_size,
            )

        if self.return_outputs:
            outputs_buffers = keras.tree.map_shape_structure(
                lambda shape: keras.ops.zeros(
                    (timesteps, batch_size) + shape, 
                    dtype=dtype
                ),
                self.cell.output_size
            )
        else:
            outputs_buffers = None

        if self.return_logs:
            logs_buffers = keras.tree.map_shape_structure(
                lambda shape: keras.ops.zeros(
                    (timesteps, batch_size) + shape, 
                    dtype=dtype
                ),
                self.cell.log_size
            )
        else:
            logs_buffers = None

        output = self.__call_dynamic(
            inputs=inputs,
            initial_state=initial_state,
            timesteps=timesteps,
            outputs_buffers=outputs_buffers,
            logs_buffers=logs_buffers,
            training=training
        )

        if self.return_outputs:
            outputs_buffers = output.pop("outputs_buffers", None)
            outputs_buffers = keras.tree.map_structure(
                lambda buffer: keras.ops.transpose(
                    x=buffer, 
                    axes=[1, 0] + list(range(2, keras.ops.ndim(buffer)))
                ), 
                outputs_buffers
            )
            output[OUTPUTS_KEY] = outputs_buffers

        if self.return_logs:
            logs_buffers = output.pop("logs_buffers", None)
            logs_buffers = keras.tree.map_structure(
                lambda buffer: keras.ops.transpose(
                    x=buffer, 
                    axes=[1, 0] + list(range(2, keras.ops.ndim(buffer)))
                ), 
                logs_buffers
            )
            output[LOGS_KEY] = logs_buffers

        return output

    def compute_output_shape(self, input_shape):
        flat_inputs_shape, _ = optree.tree_flatten(input_shape, is_leaf=is_node_a_tensor_shape)
        
        first_input_shape = flat_inputs_shape[0]
        batch_size = first_input_shape[0]
        timesteps = first_input_shape[1]

        outputs_shape = {
            STATE_KEY: keras.tree.map_shape_structure(
                lambda shape: (batch_size,) + shape, 
                self.cell.state_size
            )
        }

        if self.return_outputs:
            outputs_shape[OUTPUTS_KEY] = keras.tree.map_shape_structure(
                lambda shape: (batch_size, timesteps) + shape,
                self.cell.output_size
            )

        if self.return_logs:
            outputs_shape[LOGS_KEY] = keras.tree.map_shape_structure(
                lambda shape: (batch_size, timesteps) + shape,
                self.cell.log_size
            )

        return outputs_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "cell": keras.layers.serialize(self.cell),
            "return_outputs": self.return_outputs,
            "return_logs": self.return_logs
        })
        return config

    @classmethod
    def from_config(cls, config):
        cell_config = config.pop("cell")
        cell = keras.layers.deserialize(cell_config)
        return cls(cell=cell, **config)
