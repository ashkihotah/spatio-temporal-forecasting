from omegaconf import OmegaConf
import yaml

from hydra.utils import get_method
import numpy as np

from dvc_mlops.aim_utils.consumer import AimMetric
from dvc_mlops import StageContextManager, dvc_stage

from experiments import ROOT_DIR
from experiments.stages.test import dump_interpretations

def main(ctx: StageContextManager):
    cfg = ctx.cfg
    paths_cfg = cfg.paths
    logger = ctx.logger

    X = np.load(paths_cfg.X_path, allow_pickle=True)
    Y = np.load(paths_cfg.Y_path, allow_pickle=True)
    test_indexes = np.load(paths_cfg.test_indexes_path)
    testX, testY = X[test_indexes], Y[test_indexes]

    test_fn = get_method(cfg.test.fn)
    outputs, eval_metrics = test_fn(
        logger=logger,
        model_cfg=cfg.model,
        compile_cfg=cfg.test.compile,
        testX=testX, testY=testY,
        adj_path=paths_cfg.adj_path,
        batch_size=cfg.test.batch_size,
        scalers_path=paths_cfg.scalers_path,
        dataset_args=OmegaConf.to_container(cfg.dataset, resolve=True),

        weights_path=paths_cfg.model_path,
    )

    for metric_name, metric_value in eval_metrics.items():
        ctx.dumper.dump_metric(
            AimMetric(
                name=metric_name, value=metric_value,
                subset="test"
            )
        )

    with open(paths_cfg.test_metrics_path, "w") as f:
        yaml.dump(eval_metrics, f)
    
    dump_interpretations(logger, paths_cfg, outputs)

@dvc_stage(root_dir=ROOT_DIR)
def launcher(ctx: StageContextManager):
    main(ctx=ctx)

if __name__ == "__main__":
    launcher()