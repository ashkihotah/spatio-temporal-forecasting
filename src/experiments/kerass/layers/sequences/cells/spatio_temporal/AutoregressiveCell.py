import keras

class AutoregressiveCell:
    
    def get_cell_input_shape(self, prev_outputs_shape, current_inputs_shape):
        exogenous_features_shape, adjacency_shape = current_inputs_shape
        features_shape = prev_outputs_shape[:-1] + (
            prev_outputs_shape[-1] + exogenous_features_shape[-1],
        )
        return features_shape, adjacency_shape

    def get_cell_input(self, prev_outputs, current_inputs):
        exogenous_features, adjacency = current_inputs
        features = keras.ops.concatenate(
            [prev_outputs, exogenous_features], axis=-1
        )
        return features, adjacency