def strip_time_dim(shape):
    if shape is None:
        return None
    shape = list(shape)
    if len(shape) > 1:
        shape.pop(1)  # Remove the time axis (axis 1)
    return tuple(shape)