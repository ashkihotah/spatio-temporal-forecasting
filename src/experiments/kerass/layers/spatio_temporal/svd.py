import keras

def apply_svd(x: keras.KerasTensor, order: int):
    u, s, transposed_v = keras.ops.svd(x, full_matrices=False)
    Z = keras.ops.shape(s)[-1]
    order = keras.ops.minimum(order, Z)
    mask = keras.ops.concatenate([
        keras.ops.ones(order, dtype=s.dtype), 
        keras.ops.zeros(Z - order, dtype=s.dtype)
    ], axis=0)
    s *= mask

    expanded_s = keras.ops.expand_dims(s, axis=-2)
    scaled_u = u * expanded_s
    x_approx = keras.ops.matmul(scaled_u, transposed_v)
    return x_approx