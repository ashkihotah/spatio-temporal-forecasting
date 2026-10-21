import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

from dvc_mlops.aim_utils.consumer import AimFigure

def get_tsne_plot(
    embeds_seq: np.ndarray,
    routing_weights_seq: np.ndarray,
    global_limits: tuple,
    title_prefix: str = "Encoder Head",
) -> go.Figure:
    """Create a t-SNE plot whose point colors show the expert mixture."""
    fig = go.Figure()

    x_min, x_max, y_min, y_max = global_limits
    H = embeds_seq.shape[0]

    if embeds_seq.shape[:2] != routing_weights_seq.shape[:2]:
        raise ValueError(
            "Embedding and routing weight dimensions must match for time and node axes."
        )

    n_experts = routing_weights_seq.shape[-1]
    if n_experts > 1:
        color_sample_points = [i / (n_experts - 1) for i in range(n_experts)]
    else:
        color_sample_points = [0.5]
    expert_colors = pc.sample_colorscale("Turbo", color_sample_points)
    expert_rgb = np.array([
        tuple(int(component) for component in color[4:-1].split(","))
        for color in expert_colors
    ])
    mixed_rgb = np.clip(routing_weights_seq @ expert_rgb, 0, 255).astype(int)
    mixed_colors = [
        [f"rgb({r}, {g}, {b})" for r, g, b in timestep]
        for timestep in mixed_rgb
    ]

    node_ids = np.arange(embeds_seq.shape[1])
    fig.add_trace(
        go.Scatter(
            x=embeds_seq[0, :, 0],
            y=embeds_seq[0, :, 1],
            mode="markers",
            customdata=node_ids,
            hovertemplate="Node %{customdata}<br>x=%{x}<br>y=%{y}<extra></extra>",
            showlegend=False,
            marker=dict(color=mixed_colors[0], size=8),
        )
    )

    for expert_index, color in enumerate(expert_colors):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers", name=f"Expert {expert_index}",
                marker=dict(color=color, size=8),
            )
        )

    # Create slider steps that update coordinates and expert-mixture colors.
    steps = []
    for h in range(H):
        step = dict(
            method="update",
            args=[
                {
                    "x": [embeds_seq[h, :, 0]],
                    "y": [embeds_seq[h, :, 1]],
                    "marker.color": [mixed_colors[h]],
                },
                {"title": f"{title_prefix} {h}"},
                [0],
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
    
    fig.update_layout(
        title=f"{title_prefix} 0",
        height=600,  
        width=750,
        sliders=sliders,
        xaxis=dict(range=[x_min, x_max], title="t-SNE Dim 1"),
        yaxis=dict(range=[y_min, y_max], title="t-SNE Dim 2"),
        legend=dict(
            title=dict(text="Decoder head experts"),
            itemsizing="constant",
            traceorder="normal"
        )
    )
        
    return fig

def get_tsne_plots(deps: dict[str, str], name: str, title_prefix: str) -> list[AimFigure]:
    embeddings = np.load(deps["embeddings_path"])
    routing_weights = np.load(deps["routing_weights_path"])
    max_seq_to_plot = min(10, embeddings.shape[0])

    global_limits = (
        embeddings[..., 0].min(), embeddings[..., 0].max(),
        embeddings[..., 1].min(), embeddings[..., 1].max()
    )

    figures = []
    for seq in range(max_seq_to_plot):    

        fig_enc_tsne = get_tsne_plot(
            embeddings[seq], routing_weights[seq], global_limits,
            title_prefix=title_prefix
        )
        figures.append(AimFigure(
            name=name, 
            figure=fig_enc_tsne,
            step=seq
        ))

    return figures