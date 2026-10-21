import os
from pathlib import Path
from time import time
import logging
import yaml

import optree

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import numpy as np

import keras

from dvc_mlops.tracking.tensorflow import AsyncKerasCallback
from dvc_mlops.producing import AsyncBufferedStageDumper

from experiments.kerass.layers.scaling import ScalingLayer, KerasScaler
from experiments.kerass.metrics import ValidationMetricWrapper
from experiments.kerass.losses import DynamicLossMaskAnnealing
from experiments.kerass.callbacks import StepAnnealingCallback
from experiments.kerass.datasets import Seq2SeqDataset

from experiments.kerass.utils import model_factory
from experiments.stages.adj_utilities import localpooling_filter

from experiments.stages import get_scalers

def get_scaled_datasets(
    scalers: list[KerasScaler],
    trainX: np.ndarray,
    trainY: np.ndarray,
    
    adj_path: str | Path,
    batch_size: int,

    dataset_args: dict,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
):
    
    norm_scaling_layer = ScalingLayer(
        scalers=scalers, mode="normalize"
    )

    trainX = norm_scaling_layer(trainX).numpy() # (S, H, N, F)
    trainY = norm_scaling_layer(trainY).numpy() # (S, P, N, F)

    adj = np.load(adj_path)
    adj = localpooling_filter(adj, symmetric=True)

    val_set = None
    if validation_data is not None:
        val_x, val_y = validation_data
        norm_val_x = norm_scaling_layer(val_x).numpy()
        norm_val_y = norm_scaling_layer(val_y).numpy()
        val_set = Seq2SeqDataset(
            norm_val_x, norm_val_y, 
            adj, batch_size,
            **dataset_args
        )

    train_set = Seq2SeqDataset(
        trainX, trainY, adj, batch_size,
        **dataset_args
    )

    return train_set, val_set

def check_gpu_devices(logger: logging.Logger):
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')

    if len(gpus) == 0:
        logger.warning(f"No GPU found. Training might be slow.")
    else:
        logger.info(f"Found {len(gpus)} GPU(s). Using GPU for training.")

def get_fit_kwargs(
    fit_cfg: DictConfig,
    dumper: AsyncBufferedStageDumper,
    metrics_factory_fn,

    dump_learning_curves: bool = False,
    dlma_losses: list[keras.losses.Loss] = [],
) -> dict:
    
    fit_kwargs = instantiate(fit_cfg, _convert_="all")
    if dump_learning_curves:
        fit_kwargs["callbacks"] = fit_kwargs.get("callbacks", []) + [
            AsyncKerasCallback(
                dumper=dumper, metric_factory_fn=metrics_factory_fn
            )
        ]

    if len(dlma_losses) > 0:
        callbacks = [
            StepAnnealingCallback(
                loss_instance=loss,
                dumper=dumper,
                metric_factory_fn=metrics_factory_fn
            ) for loss in dlma_losses
        ]
        fit_kwargs["callbacks"] = fit_kwargs.get("callbacks", []) + callbacks

    return fit_kwargs

def get_compile_metrics(
    metrics: dict, target_scalers: list[KerasScaler]
) -> dict:
    for output_name, scaled_metrics in metrics.items():
        raw_metrics = [
            ValidationMetricWrapper(
                metric=metric, target_scalers=target_scalers,
                name=f"raw_{metric.name.split('scaled_')[-1]}"
            )
            for metric in scaled_metrics
        ]

        metrics[output_name] = scaled_metrics + raw_metrics

    return metrics

def get_compile_kwargs(
    compile_cfg: DictConfig,
    target_scalers: list[KerasScaler],

    min_p: int,
    max_p: int,
    epochs: int,
    batch_size: int,
    dataset_size: int,
) -> dict:
    
    loss_cfg = compile_cfg.pop("loss")
    loss_cfg_dict = OmegaConf.to_container(loss_cfg, resolve=True)

    compile_kwargs = instantiate(compile_cfg, _convert_="all")

    dlma_class = "experiments.kerass.losses.DynamicLossMaskAnnealing"

    dec_outs_cfg = loss_cfg_dict["decoder_outputs"]
    if dec_outs_cfg["_target_"] == dlma_class:
        dec_outs_cfg["min_p"] = min_p
        dec_outs_cfg["max_p"] = max_p
        if dec_outs_cfg["mode"] == "batch":
            dec_outs_cfg["total_steps"] = int(
                np.ceil(dataset_size / batch_size)
            ) * epochs
        else:
            dec_outs_cfg["total_steps"] = epochs

    loss = instantiate(loss_cfg_dict, _convert_="all")
    
    compile_kwargs["loss"] = loss

    metrics = compile_kwargs["metrics"]
    compile_kwargs["metrics"] = get_compile_metrics(
        metrics=metrics, target_scalers=target_scalers
    )
    
    return compile_kwargs

def train(
    train_cfg: DictConfig,
    dataset_args: dict,
    dumper: AsyncBufferedStageDumper,
    logger: logging.Logger,
    trainX: np.ndarray,
    trainY: np.ndarray,
    metrics_factory_fn,
    scalers_path: str | Path,
    validation_data: tuple[np.ndarray, np.ndarray] = None,
    model_cfg: DictConfig = None,
    model_path: str | Path = None,
    adj_path: str | Path = None,
) -> tuple[keras.Model, float]:
    
    keras.utils.set_random_seed(42)

    train_tensor = np.concatenate((trainX, trainY), axis=1) # (S, H+P, N, F)
    scalers: list[KerasScaler] = get_scalers(model_cfg.scalers, train_tensor)
    scalers_dicts = [scaler.to_dict() for scaler in scalers]
    trainX = train_tensor[:, :trainX.shape[1], :, :] # (S, H, N, F)
    trainY = train_tensor[:, trainX.shape[1]:, :, :] # (S, P, N, F)

    os.makedirs(os.path.dirname(scalers_path), exist_ok=True)
    with open(scalers_path, "w") as f:
        yaml.dump(scalers_dicts, f)

    if model_cfg is None:
        raise ValueError(
            "Either a `keras.Model` or a `model_cfg: DictConfig` " \
            "must be provided."
        )

    train_set, val_set = get_scaled_datasets(
        scalers=scalers,
        trainX=trainX, trainY=trainY,
        adj_path=adj_path,
        batch_size=train_cfg.fit.batch_size,
        validation_data=validation_data,
        dataset_args=dataset_args,
    )

    train_model: keras.Model = model_factory(
        model_cfg=model_cfg,
        dataset_shapes=train_set.get_shapes()
    )

    compile_kwargs = get_compile_kwargs(
        compile_cfg=train_cfg.compile,
        target_scalers=[scalers[0]],
        
        min_p=train_cfg.get("min_p", 5),
        max_p=train_cfg.get("max_p", trainY.shape[1]),
        epochs=train_cfg.fit.epochs,
        batch_size=train_cfg.fit.batch_size,
        dataset_size=trainX.shape[0],
    )

    losses = optree.tree_leaves(compile_kwargs["loss"])
    dlma_losses = [
        loss for loss in losses 
        if isinstance(loss, DynamicLossMaskAnnealing)
    ]

    train_model.compile(**compile_kwargs)

    check_gpu_devices(logger)

    model_cfg.use_exogenous_features = model_cfg.get(
        "use_exogenous_features", False
    )  

    fit_kwargs = get_fit_kwargs(
        fit_cfg=train_cfg.fit,
        dumper=dumper,
        dump_learning_curves=train_cfg.get("dump_learning_curves", False),
        metrics_factory_fn=metrics_factory_fn,
        dlma_losses=dlma_losses
    )
    
    last_time = time()
    train_model.fit(
        x=train_set,
        validation_data=val_set,
        **fit_kwargs,
        verbose=0,
        shuffle=True,
    )
    elapsed = time() - last_time
    logger.info(f"Finished training in {elapsed:.2f} seconds.")

    if model_path is not None:
        if os.path.exists(model_path):
            logger.info(
                "Weights already saved by ModelCheckpoint callback"
                f"at path: {model_path}. Skipping saving weights."
            )
        else:
            train_model.save_weights(model_path)
            logger.info(f"Saved model weights to: {model_path}")

    dumper.dump_metric(
        metrics_factory_fn(
            name="n_params", value=train_model.count_params()
        )
    )

    return train_model, elapsed