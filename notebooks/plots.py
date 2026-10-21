import numpy as np
import plotly.graph_objects as go

def plot_data(df, feature, site_id, nan_indexes=None) -> go.Figure:
    # 1. Filter and sort data by timestamp
    site_data = df[df["site_id"] == site_id].sort_values("timestamp").copy()

    # 2. Add an indicator column for filled/NaN points
    site_data["is_filled"] = site_data.index.isin(nan_indexes) if nan_indexes is not None else False

    # 3. Create a helper column to bridge boundaries:
    # To make the red line connect to the adjacent original points,
    # include original points that are immediately next to filled values.
    is_filled_mask = site_data["is_filled"]
    boundary_mask = (
        is_filled_mask | is_filled_mask.shift(1) | is_filled_mask.shift(-1)
    )

    # Replace values that aren't filled or adjacent boundaries with NaN
    # so Plotly breaks the line between separated missing segments.
    site_data["filled_segmented"] = np.where(
        boundary_mask, site_data[feature], np.nan
    )

    fig = go.Figure()

    # Base line: Plot ALL data as a continuous line (Original color)
    fig.add_trace(
        go.Scatter(
            x=site_data["timestamp"],
            y=site_data[feature],
            mode="lines",
            name="Original Values",
            line=dict(color="blue"),
            marker=dict(size=5),
        )
    )

    # Overlaid line: Plot filled segments connecting directly into original values
    fig.add_trace(
        go.Scatter(
            x=site_data["timestamp"],
            y=site_data["filled_segmented"],
            mode="lines",
            name="Filled Values",
            line=dict(color="red", width=2),
            marker=dict(size=6, color="red"),
            connectgaps=False,  # Keep gaps between distinct imputed intervals
        )
    )

    fig.update_layout(
        title=f"Site ID: {site_id} - {feature.capitalize()} Over Time",
        xaxis_title="Timestamp",
        yaxis_title=feature.capitalize(),
        hovermode="x unified",
    )

    return fig