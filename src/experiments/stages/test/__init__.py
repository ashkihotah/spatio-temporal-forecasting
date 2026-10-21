from pathlib import Path
import logging

from omegaconf import DictConfig
from sklearn.manifold import TSNE
import numpy as np

from experiments.kerass.layers.sequences.cells import HIDDEN_STATE_KEY
from experiments.kerass.layers.wrappers import ATTENTION_SCORES_KEY
from experiments.kerass.layers.sequences import (
    DECODER_LOGS_KEY, DECODER_OUTPUTS_KEY, 
    ENCODER_LOGS_KEY, ENCODER_OUTPUTS_KEY
)


def _first_nonempty_array(value):
    if isinstance(value, (list, tuple)):
        for child in value:
            result = _first_nonempty_array(child)
            if result is not None:
                return result
        return None

    array = np.asarray(value)
    if array.size > 0 and array.ndim > 0 and array.shape[-1] > 0:
        return array
    return None

def dump_tsne_embeddings(
    logger: logging.Logger, 
    tensor: np.ndarray | list[np.ndarray], 
    path: Path, 
    seed: int = 42
):

    if isinstance(tensor, list):
        tensor = tensor[-1]
    
    S_enc, H_enc, N_enc, F_enc = tensor.shape
    curr_enc = np.reshape(tensor, (S_enc*H_enc*N_enc, F_enc))
    enc_embeds = TSNE(n_components=2, perplexity=5, init="pca", random_state=seed).fit_transform(curr_enc)
    enc_embeds = np.reshape(enc_embeds, (S_enc, H_enc, N_enc, 2))

    logger.info(f"Processed tensors into shape {enc_embeds.shape} for t-SNE visualization.")

    np.save(path, enc_embeds)
    logger.info(f"Saved tensors embeddings to {path}")

def dump_interpretations(
    logger: logging.Logger, 
    paths_cfg: DictConfig, 
    outputs_dict: dict,
):
    predictions_dump_path = paths_cfg.get("decoder_preds_dump_path", None)
    if predictions_dump_path:
        np.save(predictions_dump_path, outputs_dict[DECODER_OUTPUTS_KEY])
        logger.info(f"Saved predictions to {predictions_dump_path}")

    predictions_dump_path = paths_cfg.get("encoder_preds_dump_path", None)
    if predictions_dump_path:
        enc_outputs = outputs_dict.get(ENCODER_OUTPUTS_KEY, None)
        if enc_outputs is not None:
            np.save(predictions_dump_path, enc_outputs)
            logger.info(f"Saved encoder predictions to {predictions_dump_path}")

    decoder_logs = outputs_dict[DECODER_LOGS_KEY]

    attw_dump_path = paths_cfg.get("attw_dump_path", None)
    if attw_dump_path is not None:
        attention_weights = decoder_logs.get(ATTENTION_SCORES_KEY, None)
        attention_weights = _first_nonempty_array(attention_weights)
        if attention_weights is not None:
            data_path = Path(attw_dump_path)
            np.save(data_path, attention_weights)
            logger.info(f"Saved attention weights to {data_path}")
        else:
            logger.warning(
                "Attention weights contained no active non-empty state. "
                "Skipping attention weight dump."
            )
    else:
        logger.warning("Attention weights dump path not specified. Skipping attention weight dump.")

    routing_weights_dump_path = paths_cfg.get("routing_weights_dump_path", None)
    if routing_weights_dump_path is not None:
        routing_weights = decoder_logs.get("routing_weights", None)
        if isinstance(routing_weights, (list, tuple)):
            routing_weights = routing_weights[-1]
        if routing_weights is not None:
            data_path = Path(routing_weights_dump_path)
            np.save(data_path, routing_weights)
            logger.info(f"Saved routing weights to {data_path}")
        else:
            logger.warning("Routing weights not found in predictions. Skipping routing weight dump.")
    else:
        logger.warning("Routing weights dump path not specified. Skipping routing weight dump.")

    encoder_logs = outputs_dict[ENCODER_LOGS_KEY]
    enc_hidden_states = encoder_logs.get(HIDDEN_STATE_KEY, None)
    if isinstance(enc_hidden_states, (list, tuple)):
        enc_hidden_states = enc_hidden_states[-1]
    if enc_hidden_states is not None:
        hidden_states_dump_path = paths_cfg.get("enc_hidden_states_dump_path", None)
        if hidden_states_dump_path is not None:
            np.save(hidden_states_dump_path, enc_hidden_states)
        else:
            logger.warning(
                "Hidden states dump path not specified. " \
                "Skipping hidden states dump."
            )

        # enc_embeddings_dump_path = paths_cfg.get("enc_embeddings_dump_path", None)
        # if enc_embeddings_dump_path is not None:
        #     dump_tsne_embeddings(
        #         logger, enc_hidden_states, 
        #         Path(enc_embeddings_dump_path)
        #     )
        # else:
        #     logger.warning("Encoder embeddings dump path not specified. Skipping t-SNE embedding dump.")
    else:
        logger.warning("Hidden states not found in predictions. Skipping t-SNE embedding dump.")

    dec_embeddings_dump_path = paths_cfg.get("dec_embeddings_dump_path", None)
    dec_hidden_states = decoder_logs.get(HIDDEN_STATE_KEY, None)
    if isinstance(dec_hidden_states, (list, tuple)):
        dec_hidden_states = dec_hidden_states[-1]
    if dec_embeddings_dump_path is not None:
        hidden_states_dump_path = paths_cfg.get("dec_hidden_states_dump_path", None)
        if hidden_states_dump_path is not None:
            np.save(hidden_states_dump_path, dec_hidden_states)
        else:
            logger.warning(
                "Hidden states dump path not specified. " \
                "Skipping hidden states dump."
            )
        
        # if dec_hidden_states is not None:
        #     dump_tsne_embeddings(
        #         logger, dec_hidden_states,
        #         Path(dec_embeddings_dump_path)
        #     )
        # else:
        #     logger.warning("Decoder hidden states not found in predictions. Skipping t-SNE embedding dump.")
    else:
        logger.warning("Decoder embeddings dump path not specified. Skipping t-SNE embedding dump.")
