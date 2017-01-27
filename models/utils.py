"""Everything."""


def sequential(input_tensor, *layers):
    """
    keras.models.Sequential analog for Input tensor.

    Args:
        input_tensor: instance of keras.layers.Input
        which is bare tensor that can't be used in
        models.Sequential
        layers: rest of model layers

    Returns:
        all layers sequentially applied to input tensor
    """
    last_layer = input_tensor
    for layer in layers:
        last_layer = layer(last_layer)
    return last_layer
