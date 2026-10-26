import keras

from .GCNSKIPCONN import GCNSKIPCONN
from .GraphConv import GraphConv
from .MoeGraphConv import MoeGraphConv
from .WeightedGraphConv import WeightedGraphConv
from .GAT import GAT
from .MoeGNN import MoeGNN

GNN_LAYERS = {
    "GCNSKIPCONN": GCNSKIPCONN,
    "spektral": GraphConv,
    "moe_spektral": MoeGraphConv,
    "weighted": WeightedGraphConv,
    "gat": GAT,
    "moe_gnn": MoeGNN,
}

def instantiate_gnn_layer(gnn_config: dict, **kwargs) -> keras.layers.Layer:
    gnn_type = gnn_config.pop("gnn_type")
    gnn_layer_class = GNN_LAYERS.get(gnn_type, None)
    if gnn_layer_class is None:
        raise ValueError(
            f"Invalid gnn_type '{gnn_type}'. "
            f"Supported types are: {list(GNN_LAYERS.keys())}"
        )
    return gnn_layer_class(**gnn_config, **kwargs)