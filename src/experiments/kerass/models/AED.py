from typing import TypedDict

import keras
import optree

from experiments.kerass.layers.sequences.constants import (
    DECODER_LOGS_KEY, DECODER_OUTPUTS_KEY, ENCODER_LOGS_KEY, ENCODER_OUTPUTS_KEY, LOGS_KEY, OUTPUTS_KEY, STATE_KEY
)
from experiments.kerass.layers.sequences.rnn import RNN
from experiments.kerass.layers.sequences.utils import strip_time_dim
from experiments.kerass.utils import is_node_a_tensor_shape

ATTENTION_SCORES_KEY = "attention_scores"

class AttentionConfig(TypedDict):
    attention_layers: list[keras.layers.Layer]
    query_projections: list[keras.layers.Layer]
    values_projections: list[keras.layers.Layer]
    keys_projections: list[keras.layers.Layer]
    pe_layers: list[keras.layers.Layer]
    states_mask: any
    return_attention_scores: bool

def get_attn_inputs(
    cfg: AttentionConfig, 
    queries, attn_set, index, 
    current_time_step: int
):
    query_proj = cfg["query_projections"][index]
    values_proj = cfg["values_projections"][index]
    keys_proj = cfg["keys_projections"][index]

    _, L, *other_shape = keras.ops.shape(attn_set)
    other_1_shape = (1,) * (len(other_shape) - 1)
    attn_set_pos = keras.ops.arange(L, dtype="float32")
    attn_set_pos = keras.ops.reshape(attn_set_pos, (1, L, *other_1_shape))

    L = keras.ops.shape(queries)[1]
    query_pos = keras.ops.cast(current_time_step, "float32")
    query_pos = keras.ops.reshape(query_pos, (1, L, *other_1_shape))

    pe_layer = cfg["pe_layers"][index]
    if pe_layer is not None:
        queries = pe_layer((queries, query_pos))
        keys = pe_layer((attn_set, attn_set_pos))
    else:
        keys = attn_set

    queries = query_proj(queries)
    values = values_proj(attn_set)
    keys = keys_proj(keys)

    return queries, values, keys

class AED(keras.layers.Layer):

    def __init__(
        self,
        p: int,

        encoder_cell: keras.layers.Layer,
        decoder_cell: keras.layers.Layer = None,
        attention_config: AttentionConfig = None,

        use_encoder_last_outputs_as_decoder_initial_outputs: bool = False,

        return_encoder_cell_outputs: bool = False,
        return_decoder_cell_outputs: bool = False,

        return_encoder_cell_logs: bool = False,
        return_decoder_cell_logs: bool = False,

        **kwargs
    ):
        super().__init__(**kwargs)

        self.return_encoder_cell_outputs = return_encoder_cell_outputs
        self.return_decoder_cell_outputs = return_decoder_cell_outputs
        self.return_encoder_cell_logs = return_encoder_cell_logs
        self.return_decoder_cell_logs = return_decoder_cell_logs
        self.use_encoder_last_outputs_as_decoder_initial_outputs = use_encoder_last_outputs_as_decoder_initial_outputs

        self.decoder_cell = decoder_cell
        if self.decoder_cell is None:
            self.decoder_cell = encoder_cell
        self.prediction_horizon = p
        self.attention_config = attention_config

        return_encoder_cell_outputs = return_encoder_cell_outputs or use_encoder_last_outputs_as_decoder_initial_outputs

        return_encoder_cell_logs = return_encoder_cell_logs or self.attention_config is not None
        self.encoder = RNN(
            cell=encoder_cell, 
            return_outputs=return_encoder_cell_outputs,
            return_logs=return_encoder_cell_logs,
            name="encoder"
        )

        if self.attention_config is not None:
            self.encoder.cell.return_state = True

    def _build_decoder_cell(self, input_shape):
        initial_outputs_shape = input_shape[0]
        inputs_shape = input_shape[1]

        if inputs_shape is not None:
            inputs_step_shape = keras.tree.map_shape_structure(
                strip_time_dim, inputs_shape
            )

        if not self.decoder_cell.built:
            cell_input_shape = self.decoder_cell.get_cell_input_shape(
                prev_outputs_shape=initial_outputs_shape, 
                current_inputs_shape=inputs_step_shape
            )
            self.decoder_cell.build(cell_input_shape)

    def _build_attention(self, cfg: AttentionConfig):
        self.state_size = self.encoder.cell.state_size
        keras.tree.assert_same_structure(
            self.state_size, self.decoder_cell.state_size,
        )

        state_layout_schema = keras.tree.map_shape_structure(
            lambda x: True, self.state_size
        )
        keras.tree.assert_same_paths(state_layout_schema, cfg["states_mask"])
        
        flat_mask = keras.tree.flatten(cfg["states_mask"])
        self._active_indices = [
            i for i, wanted in enumerate(flat_mask) if wanted
        ]
        _num_active = len(self._active_indices)
        
        if len(cfg["attention_layers"]) != _num_active:
            raise ValueError(
                f"Number of attention_layers ({len(cfg["attention_layers"])}) "
                f"must match the number of active states in the mask "
                f"({_num_active})."
            )

        self.flat_state_size, self.state_pytree_spec = optree.tree_flatten(
            self.state_size, is_leaf=is_node_a_tensor_shape
        )
        
        for i, idx in enumerate(self._active_indices):
            state_shape = (None,) + self.flat_state_size[idx]

            queries_shape = (None, 1) + self.flat_state_size[idx]
            attn_set_shape = (None, self.H) + self.flat_state_size[idx]

            query_proj_layer = cfg["query_projections"][i]
            if not query_proj_layer.built:
                query_proj_layer.build(queries_shape)

            keys_proj_layer = cfg["keys_projections"][i]
            if not keys_proj_layer.built:
                keys_proj_layer.build(attn_set_shape)

            values_proj_layer = cfg["values_projections"][i]
            if not values_proj_layer.built:
                values_proj_layer.build(attn_set_shape)

            attn_layer = cfg["attention_layers"][i]
            if not attn_layer.built:
                attention_shape = (
                    queries_shape, attn_set_shape, attn_set_shape
                )
                attn_layer.build(attention_shape)

            pe_layer = cfg["pe_layers"][i]
            if pe_layer is not None:
                if not pe_layer.built:
                    pe_layer.build(state_shape)

    def build(self, input_shape):
        encoder_input_shape = input_shape[0]
        decoder_inputs_shape = input_shape[1]
        decoder_initial_outputs_shape = input_shape[2]

        flat_encoder_input_shape = optree.tree_leaves(
            encoder_input_shape, is_leaf=is_node_a_tensor_shape
        )
        self.H = flat_encoder_input_shape[0][1]
        
        self.encoder.build(encoder_input_shape)
        self._build_decoder_cell((
            decoder_initial_outputs_shape,
            decoder_inputs_shape
        ))
        if self.attention_config is not None:
            self._build_attention(self.attention_config)

        super().build(input_shape)

    def _get_att_scores_shapes(
        self, cfg: AttentionConfig,
    ):
        attw_shapes = []
        for i in range(len(self.flat_state_size)):
            state_shape = self.flat_state_size[i] 
            queries_shape = (None, 1, ) + state_shape
            attn_set_shape = (None, self.H, ) + state_shape
            if i in self._active_indices:
                attw_shape = cfg["attention_layers"][i].get_attn_scores_shape(
                    queries_shape, attn_set_shape
                )
            else:
                attw_shape = state_shape[:-1] + (0,)
            attw_shapes.append(attw_shape)

        attw_shapes = optree.tree_unflatten(
            self.state_pytree_spec, attw_shapes
        )

        return attw_shapes

    def _apply_attention(
        self, 
        dec_state, enc_states,
        cfg: AttentionConfig,
        current_time_step: int,
        training=None,
    ):
        attention_scores = []
        flat_dec_states = optree.tree_leaves(dec_state)
        flat_enc_states = optree.tree_leaves(enc_states)

        active_index = 0
        for idx, dec_state in enumerate(flat_dec_states):
            if idx in self._active_indices:
                enc_state = flat_enc_states[idx]
                attention_inputs = get_attn_inputs(
                    cfg, 
                    keras.ops.expand_dims(dec_state, axis=1), 
                    enc_state, active_index, current_time_step
                )
                attn_output, scores = cfg["attention_layers"][active_index](
                    attention_inputs,
                    return_attention_scores=True,
                    training=training
                )
                attention_scores.append(keras.ops.squeeze(scores, axis=1))
                flat_dec_states[idx] = keras.ops.squeeze(attn_output, axis=1)
                active_index += 1
            else:
                # Placeholder for inactive states to maintain structure
                attention_scores.append(dec_state[..., :0])
                flat_dec_states[idx] = dec_state

        modified_states = optree.tree_unflatten(
            self.state_pytree_spec, flat_dec_states
        )
        attention_scores = optree.tree_unflatten(
            self.state_pytree_spec, attention_scores
        )
        return modified_states, attention_scores

    def _get_dynamic_fns(self, inputs, training=None):

        if self.attention_config is not None:
            inputs, enc_states = inputs
    
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

            cell_inputs = self.decoder_cell.get_cell_input(
                prev_outputs=current_outputs, 
                current_inputs=inputs_at_t
            )

            if self.attention_config is not None:
                current_state, attention_scores = self._apply_attention(
                    current_state, enc_states,
                    self.attention_config,
                    current_time_step=t,
                    training=training
                )

            outputs, next_state, logs, step_loss = self.decoder_cell(
                cell_inputs, current_state, training=training
            )

            if self.attention_config is not None:
                if self.attention_config["return_attention_scores"]:
                    logs[ATTENTION_SCORES_KEY] = attention_scores

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

            if self.return_decoder_cell_logs:
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

    def _ar_call_dynamic(
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
        if self.return_decoder_cell_logs:
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

    def ar_call(
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
            initial_states = self.decoder_cell.get_initial_state(
                batch_size=batch_size,
                dtype=dtype
            )

        outputs_buffers = keras.tree.map_shape_structure(
            lambda shape: keras.ops.zeros(
                (self.prediction_horizon, batch_size) + shape, 
                dtype=dtype
            ),
            self.decoder_cell.output_size
        )

        if self.return_decoder_cell_logs:
            logs_buffers = keras.tree.map_shape_structure(
                lambda shape: keras.ops.zeros(
                    (self.prediction_horizon, batch_size) + shape, 
                    dtype=dtype
                ),
                self.decoder_cell.log_size
            )
            if self.attention_config is not None:
                if self.attention_config["return_attention_scores"]:
                    attn_scores_shapes = self._get_att_scores_shapes(
                        self.attention_config
                    )
                    attn_scores_buffers = keras.tree.map_shape_structure(
                        lambda shape: keras.ops.zeros(
                            (self.prediction_horizon, batch_size) + shape, 
                            dtype=dtype
                        ),
                        attn_scores_shapes
                    )
                    logs_buffers[ATTENTION_SCORES_KEY] = attn_scores_buffers
        else:
            logs_buffers = None

        output = self._ar_call_dynamic(
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

        if self.return_decoder_cell_logs:
            logs_buffers = output.pop("logs_buffers", None)
            logs_buffers = keras.tree.map_structure(
                lambda buffer: keras.ops.transpose(
                    x=buffer, axes=[1, 0] + list(range(2, keras.ops.ndim(buffer)))
                ), 
                logs_buffers
            )
            output[LOGS_KEY] = logs_buffers

        return output

    def call(self, inputs, training=None):
    
        encoder_inputs = inputs[0]
        decoder_inputs = inputs[1]
        decoder_initial_outputs = inputs[2]

        enc_out = self.encoder(encoder_inputs, training=training)
        encoder_final_state = enc_out[STATE_KEY] # (B, H, ...)

        if self.use_encoder_last_outputs_as_decoder_initial_outputs:
            enc_preds = enc_out[OUTPUTS_KEY] # (B, H, ...)
            decoder_initial_outputs = enc_preds[:, -1, ...] # (B, N, ...)

        if self.attention_config is not None:
            enc_h = enc_out[LOGS_KEY][STATE_KEY] # (B, H, ...)
            decoder_inputs = (decoder_inputs, enc_h)

        dec_out = self.ar_call(
            initial_outputs=decoder_initial_outputs,
            initial_states=encoder_final_state,
            inputs=decoder_inputs,
            training=training
        )

        output_dict = {
            DECODER_OUTPUTS_KEY: dec_out[OUTPUTS_KEY]
        }
        if self.return_encoder_cell_outputs:
            output_dict[ENCODER_OUTPUTS_KEY] = enc_out[OUTPUTS_KEY]
        if self.return_decoder_cell_logs:
            output_dict[DECODER_LOGS_KEY] = dec_out[LOGS_KEY]
        if self.return_encoder_cell_logs:
            output_dict[ENCODER_LOGS_KEY] = enc_out[LOGS_KEY]

        return output_dict