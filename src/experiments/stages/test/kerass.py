from pathlib import Path
import logging
import yaml

from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np

import keras

from experiments.kerass.layers.scaling import KerasScaler, ScalingLayer
from experiments.kerass.utils import model_factory
from experiments.stages.adj_utilities import localpooling_filter
from experiments.kerass.layers.sequences import (
    DECODER_OUTPUTS_KEY, ENCODER_OUTPUTS_KEY
)
from experiments.kerass.datasets import Seq2SeqDataset

from experiments.stages.train.kerass import get_compile_metrics

def inspect_layer_tree(layer, indent=0):
    """Recursively prints sub-layers of any Keras Layer or Model."""
    prefix = "  │" * indent + ("── " if indent > 0 else "")
    
    # Calculate parameter counts
    params = sum(keras.ops.size(w) for w in layer.weights) if getattr(layer, "built", False) else "Unbuilt"
    print(f"{prefix}{layer.name} [{layer.__class__.__name__}] (Params: {params})")

    # 1. Retrieve child layers from Model (.layers) or Layer (_layers / _flatten_layers)
    sublayers = []
    if hasattr(layer, "layers") and layer.layers:
        sublayers = layer.layers
    elif hasattr(layer, "_layers") and layer._layers:
        sublayers = layer._layers

    # 2. Recursively inspect direct child layers
    for sub in sublayers:
        # Avoid printing self recursively if _layers contains self
        if sub is not layer:
            inspect_layer_tree(sub, indent + 1)

def reload_model_weights(
    model: keras.Model,
    dataset_shapes,
    logger: logging.Logger,
    weights_path: str | Path = None,
    from_model: keras.Model = None,
) -> keras.Model:

    model_inputs_shapes, model_outputs_shapes = dataset_shapes
    model.build(model_inputs_shapes)

    if from_model is not None:
        model.set_weights(from_model.get_weights())
        logger.info("Weights copied from provided model.")
    else:
        if weights_path is None:
            raise ValueError(
                "Either a `keras.Model` or a `weights_path` " \
                "must be provided."
            )
        model.load_weights(weights_path)
        logger.info(f"Weights loaded from: {weights_path}")

    return model

def get_test_set(
    scalers: list[KerasScaler],
    testX: np.ndarray, testY: np.ndarray,
    adj_path: str | Path,
    batch_size: int,
    dataset_args: dict,
) -> tuple[keras.utils.Sequence, keras.utils.Sequence]:
    
    norm_scaling_layer = ScalingLayer(
        scalers=scalers, mode="normalize"
    )

    adj = np.load(adj_path)
    adj = localpooling_filter(adj, symmetric=True)

    norm_test_x = norm_scaling_layer(testX).numpy()
    norm_test_y = norm_scaling_layer(testY).numpy()
    test_set = Seq2SeqDataset(
        norm_test_x, norm_test_y, 
        adj, batch_size,
        **dataset_args
    )

    return test_set

def test(
    logger: logging.Logger,
    model_cfg: DictConfig,
    dataset_args: dict,
    compile_cfg: DictConfig,
    testX: np.ndarray, 
    testY: np.ndarray,
    adj_path: str | Path,
    batch_size: int,
    scalers_path: str | Path,
    weights_path: Path = None,
) -> np.ndarray:
    
    keras.utils.set_random_seed(42)

    with open(scalers_path, "r") as f:
        scalers_dicts = yaml.safe_load(f)
    scalers = [instantiate(scaler_dict) for scaler_dict in scalers_dicts]
    target_scalers = [scalers[0]]

    test_set = get_test_set(
        scalers=scalers,
        testX=testX, testY=testY,
        adj_path=adj_path,
        batch_size=batch_size,
        dataset_args=dataset_args,
    )

    test_model: keras.Model = model_factory(
        model_cfg=model_cfg.copy(),
        dataset_shapes=test_set.get_shapes()
    )

    use_exogenous_features = model_cfg.get("use_exogenous_features", False) 
    test_model = reload_model_weights(
        model=test_model,
        dataset_shapes=test_set.get_shapes(),
        logger=logger,
        weights_path=weights_path
    )

    compile_kwargs = instantiate(compile_cfg, _convert_="all")
    compile_kwargs["metrics"] = get_compile_metrics(
        metrics=compile_kwargs["metrics"],
        target_scalers=target_scalers,
    )
    test_model.compile(**compile_kwargs)

    eval_metrics = test_model.evaluate(
        test_set, verbose=0, return_dict=True
    )

    model_cfg.init.return_encoder_cell_logs = True
    model_cfg.init.return_decoder_cell_logs = True
    interpreted_model: keras.Model = model_factory(
        model_cfg=model_cfg,
        dataset_shapes=test_set.get_shapes()
    )

    interpreted_model = reload_model_weights(
        model=interpreted_model,
        dataset_shapes=test_set.get_shapes(),
        logger=logger,
        weights_path=weights_path
    )

    outputs = interpreted_model.predict(test_set, verbose=0)

    scaling_layer = ScalingLayer(scalers=target_scalers, mode="denormalize")
    outputs[DECODER_OUTPUTS_KEY] = scaling_layer(
        keras.ops.squeeze(outputs[DECODER_OUTPUTS_KEY], axis=-1)
    ).numpy()
    outputs[ENCODER_OUTPUTS_KEY] = scaling_layer(
        keras.ops.squeeze(outputs[ENCODER_OUTPUTS_KEY], axis=-1)
    ).numpy()

    return outputs, eval_metrics