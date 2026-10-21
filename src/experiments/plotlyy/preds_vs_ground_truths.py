import numpy as np
import plotly.graph_objects as go
from dvc_mlops.aim_utils.consumer import AimFigure

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, node=None) -> go.Figure:
    aes = np.abs(y_true - y_pred) # absolute errors of shape (seqs, p, nodes)
    global_mae = np.mean(aes, axis=(0)) # mean across all axes except the prediction step axis (p), resulting in shape (p,)

    # for each node create a separate trace in the same plotly figure 
    # for the MAE across prediction steps
    fig = go.Figure()
    N = aes.shape[-1]
    for n in range(N):
        legend_name = f"Node {n}"

        fig.add_trace(go.Scatter(
            y=global_mae[..., n].tolist(), 
            mode='lines', 
            line=dict(width=1), 
            name=f"Node {n}",
            showlegend=True,
            legendgroup=legend_name
        ))

    # Add a mean trace across all nodes
    fig.add_trace(go.Scatter(
        y=np.mean(global_mae, axis=-1).tolist(),
        mode='lines',
        line=dict(width=2, color='black'),
        name='Mean',
        showlegend=True,
        legendgroup='Mean'
    ))

    fig.update_layout(
        title="Mean Absolute Error (MAE) Across Prediction Steps",
        xaxis_title="Prediction Step",
        yaxis_title="Mean Absolute Error (MAE)",
        xaxis=dict(tickmode='linear', dtick=1)
    )

    # create a dropdown menu to toggle visibility of individual node traces and the mean trace
    visibility_masks = {
        "Show All": [True] * (N + 1),
        "Show Only Mean": [False] * N + [True],
        "Show Only Nodes": [True] * N + [False],
        "Hide All": [False] * (N + 1)
    }

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=[
                    dict(
                        label=label,
                        method="restyle",
                        args=[{"visible": mask}]
                    ) for label, mask in visibility_masks.items()
                ]
            )
        ]
    )

    return fig

def get_prediction_plot(
    preds_seq: np.ndarray, 
    truth_seq: np.ndarray = None, 
    input_seq: np.ndarray = None,
    encoder_preds_seq: np.ndarray = None,
) -> go.Figure:
    """Creates a Plotly figure for Predictions vs Ground Truth with toggle controls."""
    fig = go.Figure()
    
    # Expand dims safely to guarantee 2-dimensional structures (Timesteps, Features)
    preds_2d = preds_seq if len(preds_seq.shape) > 1 else np.expand_dims(preds_seq, axis=1)
    
    has_truth = truth_seq is not None
    if has_truth:
        truth_2d = truth_seq if len(truth_seq.shape) > 1 else np.expand_dims(truth_seq, axis=1)
        
    has_encoder = input_seq is not None and encoder_preds_seq is not None
    if has_encoder:
        encoder_truth_2d = (
            input_seq if len(input_seq.shape) > 1
            else np.expand_dims(input_seq, axis=1)
        )
        encoder_preds_2d = (
            encoder_preds_seq if len(encoder_preds_seq.shape) > 1
            else np.expand_dims(encoder_preds_seq, axis=1)
        )
        if encoder_truth_2d.shape != encoder_preds_2d.shape:
            raise ValueError(
                "Encoder predictions and ground truth must have the same shape."
            )

    if has_encoder:
        prediction_2d = np.concatenate((encoder_preds_2d, preds_2d), axis=0)
        truth_2d = np.concatenate((encoder_truth_2d, truth_2d), axis=0)
    else:
        prediction_2d = preds_2d
        
    N = preds_2d.shape[1]
    timesteps = prediction_2d.shape[0]

    # Trace indexing calculations for the dropdown buttons
    traces_per_node = 1 + int(has_truth)
    num_node_traces = N * traces_per_node
    num_mean_traces = 1 + int(has_truth)
    total_traces = num_node_traces + num_mean_traces

    # 1. Add individual node traces grouped by node ID
    for n in range(N):
        legend_group = f"Node {n}"
        
        # Prediction trace
        fig.add_trace(go.Scatter(
            y=prediction_2d[:, n].tolist(),
            mode='lines', 
            line=dict(color='rgba(255,0,0,0.4)', width=1), 
            legendgroup=legend_group,
            name=legend_group,
            showlegend=True  # Serves as the master toggle for this node group
        ))
        
        # Truth trace
        if has_truth:
            fig.add_trace(go.Scatter(
                y=truth_2d[:, n].tolist(),
                mode='lines', 
                line=dict(color='rgba(0,0,255,0.4)', width=1), 
                legendgroup=legend_group,
                name=f"Node {n} Truth",
                showlegend=False  # Hidden from legend list, controlled by the group
            ))
            
    # 2. Add Mean traces
    fig.add_trace(
        go.Scatter(
            y=np.mean(prediction_2d, axis=-1).tolist(),
            mode='lines', 
            line=dict(color='red', width=4), 
            name='Predictions Mean',
            showlegend=True
        )
    )
    if has_truth:
        fig.add_trace(
            go.Scatter(
                y=np.mean(truth_2d, axis=-1).tolist(),
                mode='lines', 
                line=dict(color='blue', width=4), 
                name='Truth Mean',
                showlegend=True
            )
        )
    # 3. Create helper dropdown menu options
    show_all_mask = [True] * total_traces
    show_means_only_mask = [False] * num_node_traces + [True] * num_mean_traces
    show_nodes_only_mask = [True] * num_node_traces + [False] * num_mean_traces
    hide_all_mask = [False] * total_traces

    # 4. Layout configuration with button and legend positioning
    fig.update_layout(
        xaxis_title="Timesteps", 
        yaxis_title="Target",
        
        # Position the legend on the right side with a scrollbar if nodes are numerous
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            traceorder="normal"
        ),
        
        # Position the general visibility dropdown above the plot to avoid overlapping
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=[
                    dict(
                        label="Show All",
                        method="restyle",
                        args=[{"visible": show_all_mask}]
                    ),
                    dict(
                        label="Show Only Means",
                        method="restyle",
                        args=[{"visible": show_means_only_mask}]
                    ),
                    dict(
                        label="Show Only Nodes",
                        method="restyle",
                        args=[{"visible": show_nodes_only_mask}]
                    ),
                    dict(
                        label="Hide All Lines",
                        method="restyle",
                        args=[{"visible": hide_all_mask}]
                    )
                ]
            )
        ]
    )
    
    return fig

def get_prediction_plots(deps: dict[str, str], test_indexes: list[int] = None) -> list[AimFigure]:
    preds = np.load(deps["decoder_preds_path"])
    encoder_preds_path = deps.get("encoder_preds_path", None)
    if encoder_preds_path is not None:
        encoder_preds = np.load(encoder_preds_path)
    test_split = np.load(deps["test-indexes-path"])
    Y = np.load(deps["Y_path"])
    X = np.load(deps["X_path"])
    testY = Y[test_split][..., 0]
    testX = X[test_split][..., 0]

    mae_fig = evaluate(testY, preds)

    if test_indexes is not None:
        testY = testY[test_indexes]
        testX = testX[test_indexes]
        preds = preds[test_indexes]
        if encoder_preds_path is not None:
            encoder_preds = encoder_preds[test_indexes]

    max_seq_to_plot = min(10, preds.shape[0])
    
    figures = []
    for seq in range(max_seq_to_plot):
        if encoder_preds_path is not None:
            encoder_preds_seq = encoder_preds[seq]
        else:
            encoder_preds_seq = None
        fig_pred = get_prediction_plot(
            preds[seq], testY[seq], testX[seq], encoder_preds_seq
        )
        figures.append(AimFigure(
            name="Predictions vs Ground Truth", 
            figure=fig_pred,
            step=seq
        ))

    return figures + [AimFigure(
        name="Mean Absolute Error (MAE) Across Prediction Steps",
        figure=mae_fig,
        step=0
    )]
