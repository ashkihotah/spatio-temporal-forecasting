import math

from hydra.utils import get_method

from dvc_mlops import StageContextManager, dvc_stage
import numpy as np

from experiments import ROOT_DIR
from experiments.stages.test import dump_interpretations

@dvc_stage(root_dir=ROOT_DIR)
def main(ctx: StageContextManager):
    cfg = ctx.cfg
    path_cfg = cfg.paths
    train_cfg = cfg.train
    # batch_size = train_cfg.batch_size
    dumper = ctx.dumper
    logger = ctx.logger

    X = np.load(path_cfg.X_path)
    Y = np.load(path_cfg.Y_path)

    test_indexes = np.load(path_cfg.test_indexes_path)

    # if path_cfg.get("val_indexes_path", None) is not None:
    #     val_indexes = np.load(path_cfg.val_indexes_path)
    #     valX, valY = X[val_indexes], Y[val_indexes]
    #     validation_data = (valX, valY)
    # else:
    #     validation_data = None

    train_fn = get_method(cfg.train.fn)
    test_fn = get_method(cfg.test.fn)

    last_index = 0
    model = None
    preds_dict = {}
    for i, index in enumerate(test_indexes):
        logger.info(f"Training for test index: {i+1}/{len(test_indexes)}")

        # # arrotonda in eccesso per riempire il batch. potrebbe dare errore se interval + batch_size > test_indexes[0]
        # start_index = index - math.ceil(train_cfg.interval / train_cfg.batch_size) * train_cfg.batch_size
        # if start_index < 0:
        #     start_index += train_cfg.batch_size  # aggiunto per non farlo andare in negativo

        logger.info(f"Training on sequences [{last_index}, {index}]")

        # if model_params.get("dynamic_adj"):
        #     model.set_adj(
        #         create_adj(
        #             sequence=trainX[start_index:index],
        #             replicate_nodes=experiment_params.get("replicate_nodes"),
        #         )
        #     )

        model, elapsed = train_fn(
            train_cfg=train_cfg,
            model_cfg=cfg.model,
            dumper=dumper,
            logger=ctx.logger,
            metrics_factory_fn=ctx.metrics_factory_fn,

            # compiled_model=model,

            trainX=X[last_index:index], 
            trainY=Y[last_index:index], 
            validation_data=None,
        )

        if train_cfg.get("dump_training_time", False):
            dumper.dump_metric(
                ctx.metrics_factory_fn(name="training_time_seconds", value=elapsed, step=i)
            )

        pred_dict = test_fn(
            logger=logger,
            model_cfg=cfg.model,
            testX=np.expand_dims(X[index], 0),
            model=model,
        )
        preds_dict = {
            key: np.concatenate(
                [preds_dict.get(key, np.empty((0,) + value.shape[1:])), value], 
                axis=0
            )
            for key, value in pred_dict.items()
        }
        last_index = index
    
    dump_interpretations(
        logger,
        path_cfg,
        preds_dict,
        X[test_indexes]
    )

if __name__ == "__main__":
    main()