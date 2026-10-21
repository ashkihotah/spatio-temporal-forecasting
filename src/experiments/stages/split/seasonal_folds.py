from pathlib import Path
import os

import numpy as np

from dvc_mlops import StageContextManager, dvc_stage

from experiments import ROOT_DIR

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

    seasonal_folds = split_cfg.seasonal_folds

    train_seasonal_folds = seasonal_folds["train"]
    last_train_index = train_seasonal_folds[-1][1][1]
    train_indexes = np.arange(start=0, stop=last_train_index + 1, step=1)

    non_train_seasonal_folds = seasonal_folds["non_train"]
    val_indexes = []
    test_indexes = []
    for season, sequence in non_train_seasonal_folds:
        len_sequence = sequence[1] - sequence[0] + 1
        if len_sequence < 90:
            logger.warning(
                f"Sequence {sequence} in season {season} " \
                "is shorter than 90 days!"
            )

        mid_index = sequence[0] + (len_sequence // 2)

        curr_val_indexes = np.arange(
            start=sequence[0], stop=mid_index, step=1
        ).tolist()
        curr_test_indexes = np.arange(
            start=mid_index, stop=sequence[1] + 1, step=1
        ).tolist()

        val_indexes.extend(curr_val_indexes)
        test_indexes.extend(curr_test_indexes)

    val_indexes = np.array(val_indexes, dtype=np.int64)
    test_indexes = np.array(test_indexes[:-1], dtype=np.int64)

    train_size = len(train_indexes)
    val_size = len(val_indexes)
    test_size = len(test_indexes)
    total_size = train_size + val_size + test_size

    train_ratio = train_size / total_size
    val_ratio = val_size / total_size
    test_ratio = test_size / total_size

    logger.info(f"Train set size: {train_ratio * 100:.2f}% sequences.")
    logger.info(f"Validation set size: {val_ratio * 100:.2f}% sequences.")
    logger.info(f"Test set size: {test_ratio * 100:.2f}% sequences.")

    np.save(train_indexes_path, train_indexes)
    np.save(val_indexes_path, val_indexes)
    np.save(test_indexes_path, test_indexes)

if __name__ == "__main__":
    main()