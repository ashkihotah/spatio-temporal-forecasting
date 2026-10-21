from pathlib import Path

from hydra.utils import get_method
import numpy as np

from dvc_mlops import StageContextManager, dvc_stage
from omegaconf import OmegaConf

from experiments import ROOT_DIR

def main(ctx: StageContextManager):
    cfg = ctx.cfg
    path_cfg = cfg.paths
    train_cfg = cfg.train
    dumper = ctx.dumper

    X = np.load(path_cfg.X_path) # (S, H, N, F)
    Y = np.load(path_cfg.Y_path) # (S, P, N, F)

    train_indexes = np.load(path_cfg.train_indexes_path)
    trainX, trainY = X[train_indexes], Y[train_indexes]

    if path_cfg.get("val_indexes_path", None) is not None:
        val_indexes = np.load(path_cfg.val_indexes_path)
        valX, valY = X[val_indexes], Y[val_indexes]
        validation_data = (valX, valY)
    else:
        validation_data = None

    train_fn = get_method(train_cfg.fn)
    _, elapsed = train_fn(
        train_cfg=train_cfg,
        dataset_args=OmegaConf.to_container(cfg.dataset, resolve=True),
        dumper=dumper, logger=ctx.logger,
        trainX=trainX, trainY=trainY, 
        metrics_factory_fn=ctx.metrics_factory_fn,
        validation_data=validation_data,
        scalers_path=path_cfg.scalers_path,

        model_cfg=cfg.model,
        model_path=Path(cfg.paths.model_path),
        adj_path=Path(cfg.paths.adj_path),
    )

    if train_cfg.get("dump_training_time", False):
        dumper.dump_metric(
            ctx.metrics_factory_fn(name="training_time_seconds", value=elapsed)
        )

@dvc_stage(root_dir=ROOT_DIR)
def launcher(ctx: StageContextManager):
    main(ctx=ctx)

if __name__ == "__main__":
    launcher()
        