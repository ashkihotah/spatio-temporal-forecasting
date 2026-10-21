import keras

from experiments.kerass.layers.spatio_temporal.svd import apply_svd
from .PPCellAdapter import PPCellAdapter

class SVDPPAdapter(PPCellAdapter):

    def __init__(
        self,
        order: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.order = order

    def call(self, inputs, states, training=None):
        x, adj = inputs  # Unpack inputs and ignore adjacency matrix
        _, N, F = x.shape
        x_approx = apply_svd(x, self.order)
        return self.cell((x_approx, adj), states, training=training)

    def clone_with_new_cell(self, new_cell: keras.layers.Layer, **kwargs):
        return SVDPPAdapter(
            order=self.order,
            cell=new_cell,
            **kwargs
        )