import keras

from hydra.utils import instantiate
from omegaconf import DictConfig

def model_factory(
    model_cfg: DictConfig,
    dataset_shapes,
):
    model_init_cfg = model_cfg.init.copy()

    encoder_cell_cfg = model_init_cfg.pop("encoder_cell")
    decoder_cell_cfg = model_init_cfg.pop("decoder_cell", None)

    encoder_cell = instantiate(
        encoder_cell_cfg, _convert_="all",
    )
    if decoder_cell_cfg is None:
        decoder_cell = encoder_cell
    else:
        decoder_cell = instantiate(
            decoder_cell_cfg, _convert_="all",
        )

    encoder_adapter_cfg = model_cfg.get("encoder_adapter", None)
    if encoder_adapter_cfg is not None:
        encoder_cell = instantiate(
            encoder_adapter_cfg, _convert_="all",
            cell=encoder_cell,
        )

    decoder_adapter_cfg = model_cfg.get("decoder_adapter", None)
    if decoder_adapter_cfg is None:
        apply_enc_adapter_to_decoder_cell = model_cfg.get("apply_enc_adapter_to_decoder_cell", False)
        if apply_enc_adapter_to_decoder_cell:
            if encoder_adapter_cfg is not None:
                decoder_cell = encoder_cell.clone_with_new_cell(decoder_cell)
    else:
        decoder_cell = instantiate(
            decoder_adapter_cfg, _convert_="all",
            cell=decoder_cell,
        )
    
    model: keras.Model = instantiate(
        model_init_cfg, _convert_="all",
        encoder_cell=encoder_cell,
        decoder_cell=decoder_cell,
    )

    model_inputs_shapes, model_outputs_shapes = dataset_shapes
    encoder_input_shape = model_inputs_shapes[0]
    decoder_input_shape = model_inputs_shapes[1]
    decoder_initial_outputs_shape = model_inputs_shapes[2]

    enc_features_shape = encoder_input_shape[0]
    adjacency_shape = encoder_input_shape[1]
    dec_features_shape = decoder_input_shape[0]

    enc_features = keras.Input(
        shape=enc_features_shape, 
        name="raw_enc_features_input"
    )
    adjacency = keras.Input(
        shape=adjacency_shape, 
        name="adj_input"
    )
    dec_features = keras.Input(
        shape=dec_features_shape, 
        name="raw_dec_features_input"
    )
    dec_initial_outputs = keras.Input(
        shape=decoder_initial_outputs_shape, 
        name="raw_dec_initial_outputs_input"
    )

    preprocessing_layer_cfg = model_cfg.get("preprocessing_layer", None)
    if preprocessing_layer_cfg is not None:
        preprocessing_layer = instantiate(
            preprocessing_layer_cfg, _convert_="all",
        )
        pp_enc_features = preprocessing_layer((enc_features, adjacency))
    else:
        pp_enc_features = enc_features

    encoder_inputs = (pp_enc_features, adjacency)
    decoder_inputs = (dec_features, adjacency)
    output_dict = model((encoder_inputs, decoder_inputs, dec_initial_outputs))

    postprocessing_layer_cfg = model_cfg.get("postprocessing_layer", None)
    if postprocessing_layer_cfg is not None:
        postprocessing_layer = instantiate(
            postprocessing_layer_cfg, _convert_="all",
        )
        output_dict = postprocessing_layer(output_dict)

    full_model = keras.Model(
        inputs=(
            (enc_features, adjacency),
            (dec_features, adjacency),
            dec_initial_outputs
        ),
        outputs=output_dict
    )

    return full_model