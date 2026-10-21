def is_node_a_tensor_shape(x):
    is_leaf = x is None
    if not is_leaf:
        is_x_a_tuple = isinstance(x, tuple)
        all_its_elements_are_ints_or_nones = False
        if is_x_a_tuple:
            all_its_elements_are_ints_or_nones = all(
                isinstance(dim, (int, type(None))) 
                for dim in x
            )
        is_leaf = is_x_a_tuple and all_its_elements_are_ints_or_nones

    return is_leaf