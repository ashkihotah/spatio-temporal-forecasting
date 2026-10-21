import keras

from .PPCellAdapter import PPCellAdapter

class NormFFAdapter(PPCellAdapter):

    def __init__(
        self, 
        norm: keras.layers.Layer, 
        proj: keras.layers.Layer, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.norm = norm
        self.ff = proj

    def build(self, input_shape):
        features_shape, _ = input_shape
        self.norm.build(features_shape)
        self.ff.build(features_shape)
        super().build(input_shape)

    def call(self, inputs, states, training=None):
        features, adj = inputs
        x = self.norm(features)
        x = self.ff(x)
        return self.cell((x, adj), states, training=training)

    def clone_with_new_cell(self, new_cell: keras.layers.Layer, **kwargs):
        return NormFFAdapter(
            norm=self.norm,
            proj=self.ff,
            cell=new_cell,
            **kwargs
        )