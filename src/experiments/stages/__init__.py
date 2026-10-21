import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

def get_scalers(scalers_cfg: DictConfig, train_tensor: np.ndarray) -> list:
    """
    Instantiate scalers based on the provided configuration and fit them to the training tensor.

    Parameters
    ----------
    scalers_cfg : DictConfig
        Configuration for the scalers, including their classes and parameters.
    train_tensor : np.ndarray
        The training tensor used to fit the scalers.

    Returns
    -------
    list
        A list of instantiated and fitted scaler objects.
    """
    OmegaConf.set_struct(scalers_cfg, False)

    # slices of shape (S, T, N, 1) for each feature
    tensor_slices = np.split(train_tensor, train_tensor.shape[-1], axis=-1)

    scalers = [
        instantiate(scaler_cfg, _convert_="all", array=tensor_slice)
        if "from_numpy" in scaler_cfg["_target_"] 
        else instantiate(scaler_cfg, _convert_="all")
        for scaler_cfg, tensor_slice in zip(scalers_cfg, tensor_slices)
    ]
    return scalers
