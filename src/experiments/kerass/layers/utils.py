import keras

def is_single_shape(structure):
    # A shape must be a tuple or a list
    if not isinstance(structure, (tuple, list)):
        return False

    # If it is empty, it can technically represent a scalar shape ()
    if not structure:
        return True

    # If ANY element inside is a nested container (dict, list, tuple),
    # then this is a nested shape structure, NOT a single shape leaf.
    return not any(keras.tree.is_nested(x) for x in structure)

def is_namedtuple(obj):
    return isinstance(obj, tuple) and hasattr(obj, '_fields')

class BidirectionalStructuredConcatenate(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def _recursive_integrate(self, out_node, in_node):
        """Recursively matches, concatenates, or passes through tensors."""
        if out_node is None:
            return in_node
        if in_node is None:
            return out_node

        # Handle Dictionary structures (Union of keys)
        if isinstance(in_node, dict) and isinstance(out_node, dict):
            merged = {}
            all_keys = set(in_node.keys()).union(set(out_node.keys()))
            for k in all_keys:
                if k in out_node and k in in_node:
                    merged[k] = self._recursive_integrate(out_node[k], in_node[k])
                elif k in out_node:
                    merged[k] = out_node[k]
                else:
                    merged[k] = in_node[k]
            return merged

        # Handle List/Tuple structures (Max length matching)
        elif isinstance(in_node, (list, tuple)) and isinstance(out_node, (list, tuple)):
            merged = []
            max_len = max(len(in_node), len(out_node))
            for i in range(max_len):
                if i < len(out_node) and i < len(in_node):
                    merged.append(self._recursive_integrate(out_node[i], in_node[i]))
                elif i < len(out_node):
                    merged.append(out_node[i])
                else:
                    merged.append(in_node[i])
            return tuple(merged) if isinstance(in_node, tuple) else merged

        # Base Case: Both are leaf Tensors. Concatenate them along the last axis.
        elif not isinstance(in_node, (dict, list, tuple)) and not isinstance(out_node, (dict, list, tuple)):
            return keras.ops.concatenate([out_node, in_node], axis=-1)
            
        else:
            raise ValueError(
                f"Structure type mismatch at corresponding paths.\n"
                f"Output node type: {type(out_node)}\n"
                f"Input node type: {type(in_node)}"
            )

    def call(self, outputs, inputs):
        return self._recursive_integrate(outputs, inputs)

    def _recursive_shapes(self, out_node, in_node):
        """Recursively computes shapes for the matched structures."""
        if out_node is None:
            return in_node
        if in_node is None:
            return out_node

        if isinstance(in_node, dict) and isinstance(out_node, dict):
            merged = {}
            all_keys = set(in_node.keys()).union(set(out_node.keys()))
            for k in all_keys:
                if k in out_node and k in in_node:
                    merged[k] = self._recursive_shapes(out_node[k], in_node[k])
                elif k in out_node:
                    merged[k] = out_node[k]
                else:
                    merged[k] = in_node[k]
            return merged

        elif is_namedtuple(in_node) and is_namedtuple(out_node):
            merged = {}
            for field in in_node._fields:
                if hasattr(out_node, field):
                    merged[field] = self._recursive_shapes(getattr(out_node, field), getattr(in_node, field))
                else:
                    merged[field] = getattr(in_node, field)
            return type(out_node)(**merged)

        elif isinstance(in_node, (list, tuple)) and isinstance(out_node, (list, tuple)):

            if is_single_shape(in_node) and is_single_shape(out_node):
                # Both are shapes (tuples of ints), concatenate last dimension
                return out_node[:-1] + (out_node[-1] + in_node[-1],)

            merged = []
            max_len = max(len(in_node), len(out_node))
            for i in range(max_len):
                if i < len(out_node) and i < len(in_node):
                    merged.append(self._recursive_shapes(out_node[i], in_node[i]))
                elif i < len(out_node):
                    merged.append(out_node[i])
                else:
                    merged.append(in_node[i])
            return tuple(merged) if isinstance(in_node, tuple) else merged

        else:
            raise ValueError("Structure type mismatch during shape computation.")

    def compute_output_shape(self, outputs_shape, inputs_shape):
        return self._recursive_shapes(outputs_shape, inputs_shape)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
