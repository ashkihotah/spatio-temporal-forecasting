from typing import List, Any
import hashlib
import json

from omegaconf import DictConfig, OmegaConf, ListConfig

def hash_dict(d: DictConfig) -> str:
    """
    Hashes a dictionary using SHA256.

    Args:
        d (DictConfig): The dictionary to hash.

    Returns:
        str: The SHA256 hash of the dictionary.
    """

    d = OmegaConf.to_container(d, resolve=True)  # Convert DictConfig to a regular dictionary

    # remove meta fields that should not affect the hash
    d.pop("meta", None)

    # Convert the dictionary to a JSON string with sorted keys
    json_str = json.dumps(d, sort_keys=True, indent=4)
    # print(f"Hashing dictionary: {json_str}")  # Debugging line to see the JSON string being hashed
    
    # Create a SHA256 hash of the JSON string
    hash_object = hashlib.sha256(json_str.encode())
    
    # Return the hexadecimal representation of the hash
    return hash_object.hexdigest()

OmegaConf.register_new_resolver("hash_dict", hash_dict)

def concat_lists(*args: Any) -> List[Any]:
    """Concatenates an arbitrary number of lists provided as OmegaConf inputs."""
    result = []
    for arg in args:
        if arg is None:
            continue
        # Convert ListConfig or standard Python iterable to list
        if isinstance(arg, (list, ListConfig)):
            result.extend(list(arg))
        else:
            # Fallback if a single non-list element is passed
            result.append(arg)
    return list(set(sorted(result)))  # Remove duplicates while preserving order

OmegaConf.register_new_resolver("concat_lists", concat_lists)

def merge_dict_cfgs(*dicts: DictConfig) -> DictConfig:
    """
    Merges two or more DictConfigs, with later configs taking precedence over earlier ones.
    """
    merged_dict = OmegaConf.merge(*dicts)
    return merged_dict

OmegaConf.register_new_resolver(
    "merge_dict_cfgs", merge_dict_cfgs
)

def get_cmoe_cfg(
    experts: list[str],
    all_experts: list[DictConfig],
    routing_cfg: DictConfig,
) -> DictConfig:

    experts_cfgs = []
    names = set()
    file_deps = set()
    for expert_name in experts:
        expert_cfg = all_experts.get(expert_name, None)
        if expert_cfg is None:
            raise ValueError(
                f"Expert '{expert_name}' not found in all_experts"
            )
        experts_cfgs.append(expert_cfg.cfg)
        names.add(expert_cfg.name)
        file_deps.update(set(expert_cfg.file_deps))

    cmoe_cfg = DictConfig({
        "file_deps": list(file_deps),
        "cfg": {
            "_target_": "experiments.kerass.layers.sequences.cells.spatio_temporal.CMoE",
            "experts": experts_cfgs,
            **routing_cfg
        },
        "name": "CMoE-|" + "|-|".join(sorted(list(names))) + "|"
    })

    cmoe_cfg_str = OmegaConf.to_yaml(cmoe_cfg)
    cmoe_cfg_str = cmoe_cfg_str.replace("${", r"\${")
    cmoe_cfg = OmegaConf.create(cmoe_cfg_str)

    return cmoe_cfg

OmegaConf.register_new_resolver("get_cmoe_cfg", get_cmoe_cfg)

def get_n_features(n_features: int, use_time_features: bool) -> int:
    if use_time_features:
        n_features += 6
    return n_features

OmegaConf.register_new_resolver("get_n_features", get_n_features)