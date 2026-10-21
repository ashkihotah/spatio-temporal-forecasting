import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

from dvc_mlops.aim_utils.consumer import AimFigure

def get_tsne_plot(embeds_seq: np.ndarray, N: int, global_limits: tuple, title_prefix: str = "Encoder Head") -> go.Figure:
    """Creates an interactive Plotly figure for t-SNE scatter plots with a per-node legend and slider."""
    fig = go.Figure()
    
    x_min, x_max, y_min, y_max = global_limits
    H = embeds_seq.shape[0]
    
    # 1. Sample N discrete colors along the continuous 'Turbo' gradient
    if N > 1:
        color_sample_points = [n / (N - 1) for n in range(N)]
    else:
        color_sample_points = [0.5]
    node_colors = pc.sample_colorscale("Turbo", color_sample_points)

    # 2. Add one trace per node initialized to the first head (h = 0)
    for n in range(N):
        fig.add_trace(
            go.Scatter(
                x=[embeds_seq[0, n, 0]], 
                y=[embeds_seq[0, n, 1]],
                mode='markers', 
                name=f"Node {n}",
                text=[f"Node {n}"], 
                hoverinfo="text+x+y",
                showlegend=True,
                marker=dict(
                    color=node_colors[n],
                    size=8
                )
            )
        )
        
    # 3. Create slider steps that update coordinates across all N node traces
    trace_indices = list(range(N))
    steps = []
    for h in range(H):
        step = dict(
            method="update",
            args=[
                {
                    # Update each node trace's (x, y) coordinates for head `h`
                    "x": [[embeds_seq[h, n, 0]] for n in range(N)],
                    "y": [[embeds_seq[h, n, 1]] for n in range(N)]
                },
                {"title": f"{title_prefix} {h}"},
                trace_indices  # Explicitly apply updates to all N node traces
            ],  
            label=str(h)
        )
        steps.append(step)
        
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Selected: "},
        pad={"t": 50},
        steps=steps
    )]
    
    # 4. Layout configuration
    fig.update_layout(
        title=f"{title_prefix} 0",
        height=600,  
        width=750,
        sliders=sliders,
        xaxis=dict(range=[x_min, x_max], title="t-SNE Dim 1"),
        yaxis=dict(range=[y_min, y_max], title="t-SNE Dim 2"),
        legend=dict(
            title=dict(text="Nodes"),
            itemsizing="constant",
            traceorder="normal"
        )
    )
        
    return fig

def get_tsne_plots(deps: dict[str, str], name: str, title_prefix: str) -> list[AimFigure]:
    embeddings = np.load(deps["embeddings_path"])
    max_seq_to_plot = min(10, embeddings.shape[0])
    
    N = embeddings.shape[2]

    global_limits = (
        embeddings[..., 0].min(), embeddings[..., 0].max(),
        embeddings[..., 1].min(), embeddings[..., 1].max()
    )

    figures = []
    for seq in range(max_seq_to_plot):    

        fig_enc_tsne = get_tsne_plot(embeddings[seq], N, global_limits, title_prefix=title_prefix)
        figures.append(AimFigure(
            name=name, 
            figure=fig_enc_tsne,
            step=seq
        ))

    return figures