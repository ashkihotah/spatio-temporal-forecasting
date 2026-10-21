import keras
from keras import layers, ops


class GraphWaveNet(keras.Model):
    """
    Graph WaveNet (Wu et al., 2019) for Spatial-Temporal Time-Series Forecasting.
    
    Combines Dilated Causal 1D Convolutions (Gated TCN) with Spatial Graph Diffusion
    Convolutions over both fixed physical supports and a self-adaptive adjacency matrix.
    
    Expected Input Tensor Shape:  (Batch, In_Steps, Nodes, In_Features)
    Expected Output Tensor Shape: (Batch, Out_Steps, Nodes, Out_Features)
    """

    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        in_features: int = 2,
        out_features: int = 1,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers_per_block: int = 2,
        apt_embed_dim: int = 10,
        dropout: float = 0.3,
        supports: list = None,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.in_features = in_features
        self.out_features = out_features
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.layers_per_block = layers_per_block
        self.apt_embed_dim = apt_embed_dim
        self.dropout_rate = dropout
        self.supports = supports if supports is not None else []
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        # 1. Calculate receptive field to handle temporal padding
        # Receptive Field R = 1 + sum_{(b,l)} (kernel_size - 1) * 2^l
        self.receptive_field = 1
        for _ in range(self.blocks):
            additional_scope = self.kernel_size - 1
            for _ in range(self.layers_per_block):
                self.receptive_field += additional_scope
                additional_scope *= 2

        # 2. Linear projection from input features to hidden channels
        self.start_conv = layers.Conv2D(
            filters=self.residual_channels, kernel_size=(1, 1)
        )

        # 3. Instantiate Gated Temporal & Spatial Diffusion Sub-Layers
        self.filter_convs = []
        self.gate_convs = []
        self.residual_convs = []
        self.skip_convs = []
        self.bn = []
        self.gconv = []

        num_supports = len(self.supports) + (1 if self.addaptadj else 0)

        for b in range(self.blocks):
            new_dilation = 1
            for l in range(self.layers_per_block):
                # Temporal Dilated Convolution (Filter & Gate)
                self.filter_convs.append(
                    layers.Conv2D(
                        filters=self.dilation_channels,
                        kernel_size=(self.kernel_size, 1),
                        dilation_rate=(new_dilation, 1),
                        padding="valid",
                    )
                )
                self.gate_convs.append(
                    layers.Conv2D(
                        filters=self.dilation_channels,
                        kernel_size=(self.kernel_size, 1),
                        dilation_rate=(new_dilation, 1),
                        padding="valid",
                    )
                )

                # 1x1 Convolutions for Skip and Residual paths
                self.residual_convs.append(
                    layers.Conv2D(
                        filters=self.residual_channels, kernel_size=(1, 1)
                    )
                )
                self.skip_convs.append(
                    layers.Conv2D(filters=self.skip_channels, kernel_size=(1, 1))
                )
                self.bn.append(layers.BatchNormalization(axis=-1))

                # Spatial Graph Convolution (Diffusion Order K = 2)
                if self.gcn_bool:
                    # Input depth = (order * num_supports + 1) * dilation_channels
                    # Order K = 2 -> (2 * num_supports + 1)
                    in_dim = (2 * num_supports + 1) * self.dilation_channels
                    self.gconv.append(
                        layers.Conv2D(
                            filters=self.residual_channels, kernel_size=(1, 1)
                        )
                    )

                new_dilation *= 2

        # 4. Final MLP Output Projection Layers
        self.end_conv_1 = layers.Conv2D(
            filters=self.end_channels, kernel_size=(1, 1), activation="relu"
        )
        self.end_conv_2 = layers.Conv2D(
            filters=self.out_steps * self.out_features, kernel_size=(1, 1)
        )
        self.dropout = layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        """
        Creates trainable spatial node embeddings for self-adaptive graph learning.
        """
        if self.addaptadj:
            # Source Node Embedding E1: [N, d]
            self.nodevec1 = self.add_weight(
                shape=(self.num_nodes, self.apt_embed_dim),
                initializer="glorot_uniform",
                trainable=True,
                name="nodevec1",
            )
            # Target Node Embedding E2: [d, N]
            self.nodevec2 = self.add_weight(
                shape=(self.apt_embed_dim, self.num_nodes),
                initializer="glorot_uniform",
                trainable=True,
                name="nodevec2",
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Forward Pass Execution.
        
        Math Steps:
        1. Causal Padding: Left-pad temporal sequence to match receptive field.
        2. Adaptive Adjacency: A_adp = Softmax(ReLU(E1 @ E2))
        3. Gated TCN: Z = tanh(Conv_filter(X)) * sigmoid(Conv_gate(X))
        4. Spatial Diffusion: H^(k) = A @ H^(k-1) aggregated across order K=2.
        5. Skip & Residual Fusion: Sum multi-scale temporal skip representations.
        """
        in_len = ops.shape(inputs)[1]

        # ----------------------------------------------------------------------
        # STEP 1: Left Causal Temporal Padding
        # Ensures output timesteps do not leak information from future steps.
        # ----------------------------------------------------------------------
        if in_len < self.receptive_field:
            pad_len = self.receptive_field - in_len
            x = ops.pad(inputs, [[0, 0], [pad_len, 0], [0, 0], [0, 0]])
        else:
            x = inputs

        x = self.start_conv(x)
        skip_accumulator = 0.0

        # ----------------------------------------------------------------------
        # STEP 2: Compute Self-Adaptive Adjacency Matrix
        # A_adp = Softmax(ReLU(E1 @ E2)) -> [N, N]
        # Dynamically discovers spatial correlations without predefined topology.
        # ----------------------------------------------------------------------
        supports_list = list(self.supports)
        if self.addaptadj:
            spatial_scores = ops.relu(ops.matmul(self.nodevec1, self.nodevec2))
            adp = ops.softmax(spatial_scores, axis=-1)
            supports_list.append(adp)

        # ----------------------------------------------------------------------
        # STEP 3: WaveNet Stack (Dilated Convolutions + Spatial GCN)
        # ----------------------------------------------------------------------
        for i in range(self.blocks * self.layers_per_block):
            residual = x

            # A. Gated Temporal Convolution (GLU Activation)
            # filter = tanh(W_f *_d X + b_f)
            # gate   = sigmoid(W_g *_d X + b_g)
            filter_out = ops.tanh(self.filter_convs[i](residual))
            gate_out = ops.sigmoid(self.gate_convs[i](residual))
            gated_x = filter_out * gate_out

            # B. Accumulate Skip Connections across multi-scale receptive fields
            s = self.skip_convs[i](gated_x)
            
            # Match temporal slice of previous skip accumulator to current step 
            # length
            s_time_len = ops.shape(s)[1]
            if isinstance(skip_accumulator, float):
                skip_accumulator = s
            else:
                skip_accumulator = skip_accumulator[:, -s_time_len:, :, :] + s

            # C. Spatial Graph Diffusion Convolution
            if self.gcn_bool:
                # Graph Diffusion Order K = 2
                # For each support matrix A: compute [H^(0), H^(1), H^(2)] where H^(k) = A @ H^(k-1)
                gcn_outputs = [gated_x]  # H^(0) = Identity/Unaggregated state
                
                for A in supports_list:
                    # Order k = 1: Spatial aggregation over graph A
                    # Einsum: [B, T, N_in, C] x [N_in, N_out] -> [B, T, N_out, C]
                    x1 = ops.einsum("ij, btjc -> btic", A, gated_x)
                    gcn_outputs.append(x1)
                    
                    # Order k = 2: Second-hop diffusion expansion
                    x2 = ops.einsum("ij, btjc -> btic", A, x1)
                    gcn_outputs.append(x2)

                # Concatenate all spatial diffusion orders along feature dimension
                concat_gcn = ops.concatenate(gcn_outputs, axis=-1)
                x = self.gconv[i](concat_gcn)
            else:
                x = self.residual_convs[i](gated_x)

            # D. Residual Connection & Normalization
            # Align time dimension of residual to match output after unpadded dilated convs
            x = self.dropout(x, training=training)
            x_time_len = ops.shape(x)[1]
            x = x + residual[:, -x_time_len:, :, :]
            x = self.bn[i](x, training=training)

        # ----------------------------------------------------------------------
        # STEP 4: Dense MLP Projection Head
        # Projects aggregated multi-scale skip features to (Out_Steps, Out_Features)
        # ----------------------------------------------------------------------
        # Slice skip features to match the finest time step resolution
        out = ops.relu(skip_accumulator)
        out = self.end_conv_1(out)
        out = self.end_conv_2(out)  # [B, T_final, N, Out_Steps * Out_Features]

        # Extract last time slice & reshape to target prediction horizon [B, P, N, F_out]
        out = out[:, -1, :, :]  # [B, N, Out_Steps * Out_Features]
        out = ops.reshape(
            out, (-1, self.num_nodes, self.out_steps, self.out_features)
        )
        out = ops.transpose(out, (0, 2, 1, 3))  # [B, Out_Steps, Nodes, Out_Features]

        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_nodes": self.num_nodes,
                "in_steps": self.in_steps,
                "out_steps": self.out_steps,
                "in_features": self.in_features,
                "out_features": self.out_features,
                "residual_channels": self.residual_channels,
                "dilation_channels": self.dilation_channels,
                "skip_channels": self.skip_channels,
                "end_channels": self.end_channels,
                "kernel_size": self.kernel_size,
                "blocks": self.blocks,
                "layers_per_block": self.layers_per_block,
                "apt_embed_dim": self.apt_embed_dim,
                "dropout": self.dropout_rate,
                "gcn_bool": self.gcn_bool,
                "addaptadj": self.addaptadj,
            }
        )
        return config