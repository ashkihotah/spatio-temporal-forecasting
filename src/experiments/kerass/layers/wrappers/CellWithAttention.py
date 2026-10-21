import optree

import keras

from experiments.kerass.utils import is_node_a_tensor_shape
from .constants import ATTENTION_SCORES_KEY

class CellWithAttention(keras.layers.Layer):

    def __init__(
        self, 
        cell: keras.layers.Layer, 
        attention_layers: list[keras.layers.Layer],
        states_mask,
        return_attention_scores: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cell = cell
        self.attention_layers = list(attention_layers)
        self.states_mask = states_mask
        self.return_attention_scores = return_attention_scores

    def build(
        self, 
        input_shape: tuple[
            keras.KerasTensor, keras.KerasTensor, keras.KerasTensor
        ]
    ):
        cell_inputs_shape, keys_shapes, values_shapes, *temporal_shapes = input_shape
        query_time_shape = None
        key_time_shape = None
        if temporal_shapes:
            if len(temporal_shapes) != 2:
                raise ValueError(
                    "Temporal attention inputs must contain query and key times."
                )
            query_time_shape, key_time_shape = temporal_shapes

        if not self.cell.built:
            self.cell.build(cell_inputs_shape)

        # Verify states_mask topology matches cell.state_size
        state_layout_schema = keras.tree.map_shape_structure(
            lambda x: True, self.cell.state_size
        )
        keras.tree.assert_same_paths(state_layout_schema, self.states_mask)
        
        flat_mask = keras.tree.flatten(self.states_mask)
        self._active_indices = [
            i for i, wanted in enumerate(flat_mask) if wanted
        ]
        self._num_active = len(self._active_indices)
        
        if len(self.attention_layers) != self._num_active:
            raise ValueError(
                f"Number of attention_layers ({len(self.attention_layers)}) "
                f"must match the number of active states in the mask "
                f"({self._num_active})."
            )

        flat_state_schema = optree.tree_leaves(
            self.cell.state_size, 
            is_leaf=is_node_a_tensor_shape
        )
        
        for i, idx in enumerate(self._active_indices):
            state_shape = flat_state_schema[idx]
            self.history_steps = keys_shapes[i][1]  # Assuming keys_shapes[i] is (batch, history_steps, ...)

            attn_layer = self.attention_layers[i]
            if not attn_layer.built:
                attention_shape = (
                    state_shape, values_shapes[i], keys_shapes[i]
                )
                if query_time_shape is not None:
                    attention_shape += (query_time_shape, key_time_shape)
                attn_layer.build(attention_shape)

        super().build(input_shape)

    @property
    def state_size(self):
        return self.cell.state_size

    def get_initial_state(self, batch_size=None, dtype=None):
        return self.cell.get_initial_state(
            batch_size=batch_size, dtype=dtype
        )

    @property
    def log_size(self):
        log_size = self.cell.log_size
        cell_states_schema, pytree_spec = optree.tree_flatten(
            self.cell.state_size, 
            is_leaf=is_node_a_tensor_shape
        )

        if self.return_attention_scores:
            attw_shapes = []
            for i in range(len(cell_states_schema)):
                state_shape = cell_states_schema[i]
                if i in self._active_indices:
                    attw_shape = state_shape[:-1] + (self.history_steps, )
                else:
                    attw_shape = state_shape[:-1] + (0, )
                attw_shapes.append(attw_shape)

            attw_shapes = optree.tree_unflatten(pytree_spec, attw_shapes)
            log_size[ATTENTION_SCORES_KEY] = attw_shapes

        return log_size 

    @property
    def output_size(self):
        return self.cell.output_size

    def get_cell_input_shape(self, prev_outputs_shape, current_inputs_shape):
        cell_current_input_shape = current_inputs_shape[0]  
        # Assuming the first element is the cell input shape
        cell_input_shape = self.cell.get_cell_input_shape(
            prev_outputs_shape, cell_current_input_shape
        )
        # Append keys and values shapes
        return (cell_input_shape, *current_inputs_shape[1:])

    def get_cell_input(self, prev_outputs, current_inputs):
        cell_current_input = current_inputs[0]  
        # Assuming the first element is the cell input
        cell_input = self.cell.get_cell_input(prev_outputs, cell_current_input)
        # Append keys and values
        return (cell_input, *current_inputs[1:])

    def _apply_attention(
        self, states, values, keys, query_time=None, key_times=None, training=None
    ):
        attention_scores = []
        flat_states, pytree_spec = optree.tree_flatten(states)

        active_index = 0
        for idx, state in enumerate(flat_states):
            if idx in self._active_indices:
                attention_inputs = (
                    state, values[active_index], keys[active_index]
                )
                if query_time is not None:
                    attention_inputs += (query_time, key_times)
                attn_output, scores = self.attention_layers[active_index](
                    attention_inputs,
                    return_attention_scores=True,
                    training=training
                )
                attention_scores.append(scores)
                flat_states[idx] = attn_output
                active_index += 1
            else:
                # Placeholder for inactive states to maintain structure
                attention_scores.append(state[..., :0])
                flat_states[idx] = state

        modified_states = optree.tree_unflatten(pytree_spec, flat_states)
        attention_scores = optree.tree_unflatten(pytree_spec, attention_scores)
        return modified_states, attention_scores

    def call(
        self, 
        inputs: tuple[
            keras.KerasTensor, 
            list[keras.KerasTensor], 
            list[keras.KerasTensor]
        ],
        states,
        training=None
    ):

        cell_inputs, keys, values, *temporal_inputs = inputs
        query_time = None
        key_times = None
        if temporal_inputs:
            if len(temporal_inputs) != 2:
                raise ValueError(
                    "Temporal attention inputs must contain query and key times."
                )
            query_time, key_times = temporal_inputs

        modified_states, attention_scores = self._apply_attention(
            states, values, keys,
            query_time=query_time,
            key_times=key_times,
            training=training,
        )
        outputs, new_states, logs, step_loss = self.cell(
            cell_inputs, modified_states, training=training
        )

        if self.return_attention_scores:
            logs[ATTENTION_SCORES_KEY] = attention_scores

        return outputs, new_states, logs, step_loss
