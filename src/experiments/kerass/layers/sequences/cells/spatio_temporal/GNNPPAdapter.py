import keras

from .PPCellAdapter import PPCellAdapter

class GNNPPAdapter(PPCellAdapter):

    def __init__(
        self,
        gnn: keras.layers.Layer,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gnn = gnn

    def build(self, input_shape):
        x_shape, adj_shape = input_shape
        if not self.gnn.built:
            self.gnn.build((x_shape, adj_shape))
        super().build(input_shape)

    def call(self, inputs, states, training=None):
        x, adj = inputs  # Unpack inputs and ignore adjacency matrix
        _, N, F = x.shape
        x, _ = self.gnn(inputs) # (B, N, F)
        return self.cell((x, adj), states, training=training)

    def clone_with_new_cell(self, new_cell: keras.layers.Layer, **kwargs):
        return GNNPPAdapter(
            gnn=self.gnn,
            cell=new_cell,
            **kwargs
        )