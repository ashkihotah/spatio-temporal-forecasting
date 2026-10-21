import optree
import keras

from experiments.kerass.utils import is_node_a_tensor_shape

from .constants import LOGS_KEY, OUTPUTS_KEY, STATE_KEY
from .utils import strip_time_dim

class AutoRegressiveRNN(keras.layers.Layer):

    def __init__(
        self,
        cell: keras.layers.Layer,
        prediction_horizon: int,
        return_logs: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cell = cell
        self.prediction_horizon = prediction_horizon
        self.return_logs = return_logs

    def build(self, input_shape: tuple[keras.KerasTensor, keras.KerasTensor]):

        initial_outputs_shape = input_shape[0]
        inputs_shape = input_shape[1]

        if inputs_shape is not None:
            inputs_step_shape = keras.tree.map_shape_structure(
                strip_time_dim, inputs_shape
            )

        if not self.cell.built:
            cell_input_shape = self.cell.get_cell_input_shape(
                prev_outputs_shape=initial_outputs_shape, 
                current_inputs_shape=inputs_step_shape
            )
            self.cell.build(cell_input_shape)

        super().build(input_shape)

    def _get_dynamic_fns(self, inputs, training=None):

        def cond(args):
            return args["t"] < self.prediction_horizon
        
        def body(args):
            t = args["t"]
            current_state = args[STATE_KEY]
            current_outputs = args["current_outputs"]
            outputs_buffers = args["outputs_buffers"]

            if inputs is None:
                inputs_at_t = None
            else:
                inputs_at_t = keras.tree.map_structure(
                    lambda x: keras.ops.take(x, t, axis=1) 
                    if x is not None else None,
                    inputs
                )

            cell_inputs = self.cell.get_cell_input(
                prev_outputs=current_outputs, 
                current_inputs=inputs_at_t
            )

            outputs, next_state, logs, step_loss = self.cell(
                cell_inputs, current_state, training=training
            )

            indices = keras.ops.expand_dims(keras.ops.expand_dims(t, 0), 0)
            updated_outputs_buffers = keras.tree.map_structure(
                lambda buffer, val: keras.ops.scatter_update(
                    buffer, 
                    indices, 
                    keras.ops.expand_dims(val, 0)
                ),
                outputs_buffers, outputs
            )

            next_args = {
                "t": t + 1,
                STATE_KEY: next_state,
                "current_outputs": outputs,
                "outputs_buffers": updated_outputs_buffers,
                "aux_loss": args["aux_loss"] + step_loss,
            }

            if self.return_logs:
                logs_buffers = args["logs_buffers"]
                indices = keras.ops.expand_dims(keras.ops.expand_dims(t, 0), 0)
                updated_logs_buffers = keras.tree.map_structure(
                    lambda buffer, val: keras.ops.scatter_update(
                        buffer,
                        indices,
                        keras.ops.expand_dims(val, 0)
                    ) if keras.ops.shape(val)[-1] > 0 else buffer,
                    logs_buffers, logs
                )
                next_args["logs_buffers"] = updated_logs_buffers

            return next_args

        return cond, body

    def __call_dynamic(
        self,
        initial_outputs,
        initial_states,
        outputs_buffers,
        logs_buffers=None,
        inputs=None,
        training=None
    ):

        cond_fn, body_fn = self._get_dynamic_fns(inputs, training=training)
        
        loop_vars = {
            "t": keras.ops.cast(0, dtype="int32"),
            "current_outputs": initial_outputs,
            STATE_KEY: initial_states,
            "outputs_buffers": outputs_buffers,
            "aux_loss": keras.ops.zeros((), dtype=keras.ops.dtype(
                keras.tree.flatten(initial_outputs)[0]
            )),
        }
        if self.return_logs:
            loop_vars["logs_buffers"] = logs_buffers

        output = keras.ops.while_loop(
            cond_fn, body_fn, loop_vars, 
            maximum_iterations=self.prediction_horizon
        )

        output.pop("t", None)
        aux_loss = output.pop("aux_loss")
        current_outputs = output.pop("current_outputs")
        norm_aux_loss = aux_loss / keras.ops.cast(
            self.prediction_horizon, dtype=keras.ops.dtype(aux_loss)
        )
        self.add_loss(norm_aux_loss)
        return output

    def call(
        self, 
        initial_outputs,
        initial_states=None,
        inputs=None,
        training=None
    ):
        flat_initial = keras.tree.flatten(initial_outputs)
        batch_size = keras.ops.shape(flat_initial[0])[0]
        dtype = keras.ops.dtype(flat_initial[0])

        if initial_states is None:
            initial_states = self.cell.get_initial_state(
                batch_size=batch_size,
                dtype=dtype
            )

        outputs_buffers = keras.tree.map_shape_structure(
            lambda shape: keras.ops.zeros(
                (self.prediction_horizon, batch_size) + shape, 
                dtype=dtype
            ),
            self.cell.output_size
        )

        if self.return_logs:
            logs_buffers = keras.tree.map_shape_structure(
                lambda shape: keras.ops.zeros(
                    (self.prediction_horizon, batch_size) + shape, 
                    dtype=dtype
                ),
                self.cell.log_size
            )
        else:
            logs_buffers = None

        output = self.__call_dynamic(
            initial_outputs=initial_outputs,
            initial_states=initial_states,
            outputs_buffers=outputs_buffers,
            logs_buffers=logs_buffers,
            inputs=inputs,
            training=training
        )

        outputs_buffers = output.pop("outputs_buffers", None)
        outputs_buffers = keras.tree.map_structure(
            lambda buffer: keras.ops.transpose(
                x=buffer, axes=[1, 0] + list(range(2, keras.ops.ndim(buffer)))
            ), 
            outputs_buffers
        )
        output[OUTPUTS_KEY] = outputs_buffers

        if self.return_logs:
            logs_buffers = output.pop("logs_buffers", None)
            logs_buffers = keras.tree.map_structure(
                lambda buffer: keras.ops.transpose(
                    x=buffer, axes=[1, 0] + list(range(2, keras.ops.ndim(buffer)))
                ), 
                logs_buffers
            )
            output[LOGS_KEY] = logs_buffers

        return output

    def compute_output_shape(self, input_shape):
        initial_outputs_shape = input_shape[0]

        flat_initial_outputs_shape, _ = optree.tree_flatten(
            initial_outputs_shape, is_leaf=is_node_a_tensor_shape
        )

        batch_size = flat_initial_outputs_shape[0][0]

        outputs_shape = {
            STATE_KEY: keras.tree.map_shape_structure(
                lambda shape: (batch_size,) + shape, 
                self.cell.state_size
            ),
            OUTPUTS_KEY: keras.tree.map_shape_structure(
                lambda shape: (batch_size, self.prediction_horizon) + shape,
                self.cell.output_size
            )
        }

        if self.return_logs:
            outputs_shape[LOGS_KEY] = keras.tree.map_shape_structure(
                lambda shape: (batch_size, self.prediction_horizon) + shape,
                self.cell.log_size
            )

        return outputs_shape
