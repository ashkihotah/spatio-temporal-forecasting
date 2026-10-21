from pathlib import Path
import logging
import math
import os

import numpy as np

from sklearn.model_selection import train_test_split

from dvc_mlops import StageContextManager, dvc_stage

from experiments import ROOT_DIR

'''
In time-series forecasting pipelines, you often have secondary tasks that do require looking further back than h:
1. **Visualization**: Plotting scripts often draw the prediction (y) alongside a much longer visual context (e.g., "show me the last 30 days of data leading up to this prediction"). If sequence 0 were in the test set, attempting to plot data before its x window would cause an index out-of-bounds error.
2. **Evaluation Baselines**: If you evaluate the test set against classic statistical models (like ARIMA or Historical Average) alongside your neural network, those models often require a much longer continuous history to initialize than the h steps your neural network uses.
By forcing the first buffer sequences into the training set, the code guarantees that any sequence randomly selected for the test set sits deep enough in the timeline to safely extract larger contextual windows when needed.
'''
def split(logger: logging.Logger, sequences: int, train_ratio: float, val_ratio: float, test_ratio: float, buffer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequence indices into train and test sets, strictly preserving an initial buffer.

    Parameters
    ----------
    sequences : int
        Total number of sequences available.
    train_ratio : float
        Ratio of the data to use for training.
    val_ratio : float
        Ratio of the data to use for validation.
    test_ratio : float
        Ratio of the data to use for testing.
    buffer : int
        Number of initial sequences forced into the training set.

    Returns
    -------
    train_indexes : np.ndarray
        Indices for the training set.
    val_indexes : np.ndarray
        Indices for the validation set.
    test_indexes : np.ndarray
        Indices for the test set.
    """
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("train_ratio, val_ratio and test_ratio must sum to 1.")

    if train_ratio <= 0:
        raise ValueError("train_ratio must be greater than 0.")

    if test_ratio < 0:
        raise ValueError("test_ratio must be greater than or equal to 0.")
    
    if val_ratio < 0:
        raise ValueError("val_ratio must be greater than or equal to 0.")
    
    logger.info(
        f"Calculating training, validation and test splits {train_ratio}, {val_ratio}, {test_ratio}..."
    )
    logger.info("Shuffling data with seed 42.")
    
    testable_indexes = np.arange(buffer, sequences)
    
    # Split testable indices: (1 - test_ratio) for train, test_ratio for test
    non_test_indexes, test_indexes = train_test_split(
        testable_indexes,
        test_size=test_ratio,
        random_state=42
    )

    # Second split: separate validation out of the non-test dataset
    if val_ratio > 0:
        # Calculate validation size relative to the remaining non-test data
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        non_test_val_indexes, val_indexes = train_test_split(
            non_test_indexes,
            test_size=relative_val_ratio,
            random_state=42
        )
    else:
        non_test_val_indexes = non_test_indexes
        val_indexes = None
    
    # Buffer indices always go to training
    train_indexes = np.concatenate([np.arange(buffer), non_test_val_indexes])
    
    return train_indexes, val_indexes, test_indexes

def split_chronological(
    logger: logging.Logger, 
    sequences: int, 
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float,
    h: int,
    p: int,
    stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits sequence indices chronologically into contiguous blocks,
    introducing a safety gap between splits to prevent overlapping window leakage.
    """
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.")

    logger.info(f"Applying Static Paradigm: Chronological block split ({train_ratio}/{val_ratio}/{test_ratio})")

    # 1. Calculate raw split boundaries based on index counts
    end_train_raw = int(sequences * train_ratio)
    end_val_raw = int(sequences * (train_ratio + val_ratio))

    # 2. Calculate the leakage gap size in terms of sequences.
    # Two windows overlap if they share raw data points. A full window spans h + p steps.
    gap = int(math.ceil((h + p) / stride))
    logger.info(f"To prevent overlapping window leakage, a gap of {gap} sequences will be dropped between splits.")

    # 3. Carve out contiguous index ranges, dropping the safety gaps
    train_indexes = np.arange(0, end_train_raw - gap)
    
    if val_ratio > 0:
        val_indexes = np.arange(end_train_raw, end_val_raw - gap)
    else:
        val_indexes = np.array([], dtype=np.int64)
        
    test_indexes = np.arange(end_val_raw, sequences)

    return train_indexes, val_indexes, test_indexes

@dvc_stage(root_dir=ROOT_DIR)
def main(ctx: StageContextManager):
    logger = ctx.logger
    cfg = ctx.cfg
    paths_cfg = cfg.paths
    split_cfg = cfg.split
    train_indexes_path = Path(paths_cfg.train_indexes_path)
    test_indexes_path = Path(paths_cfg.test_indexes_path)
    val_indexes_path = Path(paths_cfg.val_indexes_path)
    os.makedirs(train_indexes_path.parent, exist_ok=True)
    os.makedirs(test_indexes_path.parent, exist_ok=True)
    os.makedirs(val_indexes_path.parent, exist_ok=True)

    X = np.load(paths_cfg.X_path)

    sequences = X.shape[0]
    train_indexes, val_indexes, test_indexes = split_chronological(
        logger,
        sequences,
        split_cfg.train_ratio,
        split_cfg.val_ratio,
        split_cfg.test_ratio,
        # split_cfg.buffer,
        cfg.prepare.h,
        cfg.prepare.p,
        cfg.prepare.stride
    )

    logger.info(f"Train set size: {len(train_indexes) / sequences * 100:.2f}% sequences.")
    logger.info(f"Validation set size: {len(val_indexes) / sequences * 100:.2f}% sequences.")
    logger.info(f"Test set size: {len(test_indexes) / sequences * 100:.2f}% sequences.")

    # # Se ALLOW_PARTIAL_BATCHES == False, tronca gli ultimi indici per far sì che il totale sia un multiplo di BATCH_SIZE
    # if not split_cfg.allow_partial_batches:
    #     logger.info(
    #         f"Making number of samples a multiple of batch size {split_cfg.batch_size}..."
    #     )
    #     train_indexes = train_indexes[
    #         : len(train_indexes)
    #         // split_cfg.batch_size
    #         * split_cfg.batch_size
    #     ]
    #     if val_indexes is not None:
    #         val_indexes = val_indexes[
    #             : len(val_indexes)
    #             // split_cfg.batch_size
    #             * split_cfg.batch_size
    #         ]
    #     test_indexes = test_indexes[
    #         : len(test_indexes)
    #         // split_cfg.batch_size
    #         * split_cfg.batch_size
    #     ]
    # else:
    #     logger.info("Skipped truncation of partial batches.")

    train_indexes.sort()
    np.save(
        train_indexes_path,
        train_indexes,
    )
    if test_indexes is not None:
        test_indexes.sort()
        np.save(
            test_indexes_path,
            test_indexes,
        )
    if val_indexes is not None:
        val_indexes.sort()
        np.save(
            val_indexes_path,
            val_indexes,
        )

if __name__ == "__main__":
    main()