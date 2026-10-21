from pathlib import Path
import logging
import os

import numpy as np

from dvc_mlops import StageContextManager, dvc_stage

from experiments import ROOT_DIR

def obs2seqs(logger: logging.Logger, data: np.ndarray, h: int, p: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert continuous time-series data into overlapping sequences (windows).

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (obs, nodes, features) representing raw time-series.
    h : int
        Number of history steps (length of input 'x').
    p : int
        Number of prediction steps (length of target 'y').
    stride : int
        Number of steps the sliding window moves forward.

    Returns
    -------
    x : np.ndarray
        Input sequences of shape (seqs, h, nodes, features).
    y : np.ndarray
        Target sequences of shape (seqs, p, nodes, features).
    """
    
    obs, nodes, features = data.shape
    
    # Warn if data at the very end is truncated because the remaining
    # "runway" (obs - h - p) isn't perfectly divisible by the stride.
    trailing_dropped = (obs - h - p) % stride
    if trailing_dropped != 0:
        logger.warning(
            "(obs - h - p) is not perfectly divisible by stride. "
            f"The last {trailing_dropped} observation(s) in the dataset " \
            "will be completely ignored."
        )

    # Warn if the stride is larger than the window, 
    # causing intermediate data to be skipped.
    if stride > (h + p):
        gap_size = stride - (h + p)
        logger.warning(
            f"Stride ({stride}) is larger than the window size ({h + p}). "
            f"Every time the window slides, {gap_size} observation(s) " \
            "will be completely skipped."
        )

    # Total valid sliding windows that fit without going out of bounds
    seqs = (obs - h - p) // stride + 1
    x = np.zeros((seqs, h, nodes, features))
    y = np.zeros((seqs, p, nodes, features))
    for i in range(seqs):
        x[i] = data[i * stride: i * stride + h]
        y[i] = data[i * stride + h: i * stride + h + p]

    return x, y

@dvc_stage(root_dir=ROOT_DIR)
def main(ctx: StageContextManager):
    cfg = ctx.cfg
    logger = ctx.logger
    paths_cfg = cfg.paths
    prepare_cfg = cfg.prepare
    paths_cfg.dataset_path = Path(paths_cfg.dataset_path)
    paths_cfg.X_path = Path(paths_cfg.X_path)
    paths_cfg.Y_path = Path(paths_cfg.Y_path)
    # paths_cfg.adj_path = Path(paths_cfg.adj_path)

    data = np.load(paths_cfg.dataset_path, allow_pickle=True)["data"]
    replicate_nodes = prepare_cfg.replicate_nodes

    if replicate_nodes > 0:
        logger.info(f"Replicating nodes {replicate_nodes} times.")
        data = np.repeat(data, replicate_nodes, axis=1)
        
    if prepare_cfg.mini_dataset:
        data = data[
            : len(data) / prepare_cfg.h // 10 * prepare_cfg.h,
            :,
            :,
        ]

    logger.info("Data loaded with shape " + str(data.shape) + ".")

    X, Y = obs2seqs(
        logger,
        data,
        prepare_cfg.h,
        prepare_cfg.p,
        prepare_cfg.stride,
    )
    logger.info(f"Data converted to sequences with shape {X.shape}.") # (S,T,N,F)
    
    os.makedirs(paths_cfg.X_path.parent, exist_ok=True)
    os.makedirs(paths_cfg.Y_path.parent, exist_ok=True)
    np.save(paths_cfg.X_path, X)
    np.save(paths_cfg.Y_path, Y)

if __name__ == "__main__":        
    main()