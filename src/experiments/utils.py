def dict_values_to_list(**kwargs) -> list:
    """Takes keyword arguments and returns their values as a list."""
    # first sort by key to ensure consistent order, then return the values as a list
    return [value for key, value in sorted(kwargs.items())]