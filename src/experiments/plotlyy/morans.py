import plotly.graph_objects as go
import numpy as np

from dvc_mlops.aim_utils.consumer import AimFigure


def vectorized_morans_i(
    x: np.ndarray, 
    adj: np.ndarray, 
    epsilon: float = 1e-3
) -> np.ndarray:
    """
    Vectorized Moran's I computation using pure NumPy.

    Args:
        x: np.ndarray of shape (B, N, F) representing node features across batches.
        adj: np.ndarray of shape (N, N) representing the spatial adjacency matrix.
        epsilon: Small constant to prevent division by zero.

    Returns:
        np.ndarray of shape (B,) containing Moran's I values for each batch item.
    """
    # 0. Ensure inputs are float NumPy arrays
    x = np.asarray(x, dtype=np.float64)
    adj = np.asarray(adj, dtype=np.float64)
    
    # 1. Fetch node count N
    num_nodes = float(x.shape[1])
    
    # 2. Calculate deviations from spatial node averages: (B, N, F)
    x_mean = np.mean(x, axis=1, keepdims=True)  # (B, 1, F)
    deviation = x - x_mean                      # (B, N, F)
    
    # 3. Vectorized Denominator: Sum of squared deviations across nodes -> (B, F)
    denominator = np.sum(np.square(deviation), axis=1)
    
    # 4. Vectorized Numerator
    # Permute deviation to (B, F, N) to align with (N, N) adjacency matrix
    dev_perm = np.transpose(deviation, (0, 2, 1))  # (B, F, N)
    
    # Matrix multiplication broadcasts (B, F, N) @ (N, N) -> (B, F, N)
    spatial_weighted_dev = dev_perm @ adj
    
    # Element-wise product and reduction across nodes -> (B, F)
    numerator = np.sum(spatial_weighted_dev * dev_perm, axis=2)
    
    # 5. Global normalizer scaling factor: N / W (where W is sum of all weights)
    total_weights = np.sum(adj) + 1e-7
    normalizer = num_nodes / total_weights
    
    # 6. Final calculation and reduction across features -> (B,)
    i_per_feature = normalizer * (numerator / (denominator + epsilon))
    return np.mean(i_per_feature, axis=1)

def get_morans_plot(deps: dict[str, str], name: str) -> list[AimFigure]:
    hidden_states = np.load(deps["hidden_states"])
    adj = np.load(deps["adj_path"])

    H = hidden_states.shape[1]
    coeffs = []
    for h in range(H):
        coeff = vectorized_morans_i(hidden_states[:, h], adj)
        coeffs.append(coeff)
    y = np.stack(coeffs, axis=0).transpose()

    fig = go.Figure()
    S = y.shape[0]
    for s in range(S):
        current_coeff = y[s]
        fig.add_trace(go.Scatter(
            y=current_coeff.tolist(),
            mode="lines",
            line=dict(width=1),
            opacity=0.25,
            showlegend=False,
        ))
    fig.update_layout(
        xaxis_title="Timesteps",
        yaxis_title="Spatial Autocorrelation"
    )

    return [AimFigure(figure=fig, name=name)]