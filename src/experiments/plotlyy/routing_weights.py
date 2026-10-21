import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dvc_mlops.aim_utils.consumer import AimFigure

def get_routing_weights_plot(routing_weights_seq: np.ndarray) -> go.Figure:
    """Creates a Plotly subplots figure for Routing Weights Heatmaps with a slider."""
    P = routing_weights_seq.shape[0] # // 2

    fig = go.Figure()

    for p in range(P):
        raw = routing_weights_seq[p]
        
        # Only set the traces for the first step to be visible initially
        is_visible = (p == 0)
        
        fig.add_trace(
            go.Heatmap(
                z=raw.tolist(), 
                showscale=False, 
                visible=is_visible
            )
        )
        
    # Create the slider steps
    steps = []
    for p in range(P):
        # Create a boolean visibility array for all traces: [False, False, False, ...]
        step_visibility = [False] * P
        
        # Enable visibility only for the current step `p`
        step_visibility[p] = True
        
        step = dict(
            method="update",
            args=[{"visible": step_visibility}],  # Update trace visibility
            label=str(p)
        )
        steps.append(step)
        
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Prediction Step: "},
        pad={"t": 50},  # Add padding to avoid overlapping the plots
        steps=steps
    )]
    
    # Adjust layout to reflect 1 row and include the slider
    fig.update_layout(
        # height=400,  # Reduced height since it's only 1 row now
        # width=900,
        sliders=sliders,
        xaxis_title="Experts",
        yaxis_title="Nodes"
    )
    
    # # Link axes for the 3 columns
    # for i in range(1, 4):
    #     fig.layout[f'yaxis{i}'].scaleanchor = f'x{i}'
        
    return fig

def get_routing_weights_plots(deps: dict[str, str]) -> list[AimFigure]:
    routing_weights = np.load(deps["routing_weights_path"])
    max_seq_to_plot = min(10, routing_weights.shape[0])
    
    figures = []
    for seq in range(max_seq_to_plot):
        fig_att = get_routing_weights_plot(routing_weights[seq])
        figures.append(AimFigure(
            name="Routing Weights Heatmaps", 
            figure=fig_att,
            step=seq
        ))

    return figures
