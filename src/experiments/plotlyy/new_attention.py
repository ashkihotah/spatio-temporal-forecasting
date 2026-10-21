import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dvc_mlops.aim_utils.consumer import AimFigure

def get_attention_plot(
    attw_seq: np.ndarray, 
    avg_same_seq: np.ndarray, 
    avg_all_seqs: np.ndarray
) -> go.Figure:
    """Plot decoder-time by encoder-time attention for one input sequence.

    ``attw_seq`` has shape ``(P, N, H)``. A node slider selects the node
    whose complete ``(P, H)`` attention matrix is displayed.
    """
    P, N, H = attw_seq.shape
    if avg_same_seq.shape != (N, H) or avg_all_seqs.shape != (N, H):
        raise ValueError(
            "Attention means must have shape (nodes, encoder_steps)."
        )

    seq_delta = attw_seq - avg_same_seq[None, :, :]
    all_delta = attw_seq - avg_all_seqs[None, :, :]
    delta_limit = max(
        float(np.max(np.abs(seq_delta))),
        float(np.max(np.abs(all_delta))),
        np.finfo(np.float32).eps,
    )

    titles = ["Attention vs Sequence Mean", "Attention vs Dataset Mean", "Raw Attention"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles)

    for node in range(N):
        visible = node == 0
        heatmaps = [
            go.Heatmap(
                z=seq_delta[:, node, :].tolist(),
                zmin=-delta_limit,
                zmax=delta_limit,
                colorscale="RdBu",
                showscale=False,
                visible=visible,
                hovertemplate="decoder=%{y}<br>encoder=%{x}<br>delta=%{z:.4f}<extra></extra>",
            ),
            go.Heatmap(
                z=all_delta[:, node, :].tolist(),
                zmin=-delta_limit,
                zmax=delta_limit,
                colorscale="RdBu",
                showscale=False,
                visible=visible,
                hovertemplate="decoder=%{y}<br>encoder=%{x}<br>delta=%{z:.4f}<extra></extra>",
            ),
            go.Heatmap(
                z=attw_seq[:, node, :].tolist(),
                zmin=0,
                zmax=float(np.max(attw_seq)),
                colorscale="Viridis",
                showscale=False,
                visible=visible,
                hovertemplate="decoder=%{y}<br>encoder=%{x}<br>weight=%{z:.4f}<extra></extra>",
            ),
        ]
        for column, heatmap in enumerate(heatmaps, start=1):
            fig.add_trace(heatmap, row=1, col=column)

    steps = []
    for node in range(N):
        step_visibility = [False] * (N * 3)
        step_visibility[node * 3:node * 3 + 3] = [True, True, True]
        step = dict(
            method="update",
            args=[{"visible": step_visibility}],
            label=str(node),
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Node: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        height=500,
        width=900,
        sliders=sliders,
        margin={"t": 90, "b": 90},
    )

    for i in range(1, 4):
        fig.layout[f"xaxis{i}"].title = "Encoder step"
        fig.layout[f"yaxis{i}"].title = "Decoder step"
        fig.layout[f'yaxis{i}'].scaleanchor = f'x{i}'

    return fig

def get_attention_plots(deps: dict[str, str]) -> list[AimFigure]:
    attw = np.load(deps["attw_path"])
    if attw.ndim != 4:
        raise ValueError(
            "Attention weights must have shape (samples, decoder_steps, "
            "nodes, encoder_steps)."
        )
    if 0 in attw.shape:
        raise ValueError(
            f"Attention weights contain an empty axis: shape={attw.shape}. "
            "Check states_mask and attention-layer configuration."
        )

    avgAllSeqs = np.mean(attw, axis=(0, 1))
    max_seq_to_plot = min(10, attw.shape[0])
    
    figures = []
    for seq in range(max_seq_to_plot):
        avgSameSeq = np.mean(attw[seq, :], axis=0)
        fig_att = get_attention_plot(attw[seq], avgSameSeq, avgAllSeqs)
        figures.append(AimFigure(
            name="Attention Heatmaps", 
            figure=fig_att,
            step=seq
        ))

    return figures
