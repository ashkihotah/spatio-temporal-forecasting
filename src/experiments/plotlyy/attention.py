import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dvc_mlops.aim_utils.consumer import AimFigure

def get_attention_plot(
    attw_seq: np.ndarray, 
    avg_same_seq: np.ndarray, 
    avg_all_seqs: np.ndarray
) -> go.Figure:
    """Creates a Plotly subplots figure for Attention Heatmaps with a slider."""
    P = attw_seq.shape[0] # // 2
    
    # We now only need 1 row and 3 columns
    titles = ["Delta from Seq Avg", "Delta from All Avg", "Raw Attention"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles)
    
    for p in range(P):
        raw = attw_seq[p]
        
        # Only set the traces for the first step to be visible initially
        is_visible = (p == 0)
        
        # Add the 3 heatmaps sequentially to the same 1x3 grid
        fig.add_trace(go.Heatmap(z=(raw - avg_same_seq).tolist(), showscale=False, visible=is_visible), row=1, col=1)
        fig.add_trace(go.Heatmap(z=(raw - avg_all_seqs).tolist(), showscale=False, visible=is_visible), row=1, col=2)
        fig.add_trace(go.Heatmap(z=raw.tolist(), showscale=False, visible=is_visible), row=1, col=3)
        
    # Create the slider steps
    steps = []
    for p in range(P):
        # Create a boolean visibility array for all traces: [False, False, False, ...]
        step_visibility = [False] * (P * 3)
        
        # Enable visibility only for the 3 traces corresponding to the current step `p`
        step_visibility[p * 3] = True
        step_visibility[p * 3 + 1] = True
        step_visibility[p * 3 + 2] = True
        
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
        height=400,  # Reduced height since it's only 1 row now
        width=900,
        sliders=sliders
    )
    
    # Link axes for the 3 columns
    for i in range(1, 4):
        fig.layout[f'yaxis{i}'].scaleanchor = f'x{i}'
        
    return fig

def get_attention_plots(deps: dict[str, str]) -> list[AimFigure]:
    attw = np.load(deps["attw_path"])
    avgAllSeqs = np.mean(attw, axis=(0,1)) # b h n f
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
