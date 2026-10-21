from pathlib import Path
from typing import Dict
import os

import numpy as np
import pandas as pd

from dvc_mlops import StageContextManager, dvc_stage

from experiments import ROOT_DIR
from experiments.stages.preprocess import preprocess_time_features

# Mapping for the 16 cardinal directions to angles in degrees
WD_ANGLES: Dict[str, float] = {
    'N': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
    'E': 90.0, 'ESE': 112.5, 'SE': 135.0, 'SSE': 157.5,
    'S': 180.0, 'SSW': 202.5, 'SW': 225.0, 'WSW': 247.5,
    'W': 270.0, 'WNW': 292.5, 'NW': 315.0, 'NNW': 337.5
}

def gemini_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    
    # -------------------------------------------------------------------------
    # 0. Handle Missing Values (Imputation)
    # -------------------------------------------------------------------------
    # Interpolate numeric features forward/backward to remove NaNs cleanly
    num_cols = data.select_dtypes(include=['float']).columns
    data[num_cols] = data[num_cols].ffill().bfill()
    
    
    # Fill categorical wind direction NaNs with most frequent value (mode)
    if "wd" in data.columns and data["wd"].isnull().any():
        data["wd"] = data["wd"].fillna(data["wd"].mode()[0])

    # -------------------------------------------------------------------------
    # Extreme Zero-Inflated Sparse Variable (RAIN Decomposition)
    '''
                      ┌──> Step 1a: Binary Indicator (0 or 1) ──────────┐
    Raw RAIN ──> Split│                                                 ├──> Concatenate
                      └──> Step 1b: Log-plus-one (Magnitude if > 0) ────┘
    '''
    # -------------------------------------------------------------------------
    rain_raw = data["RAIN"].values
    
    # Step 1a: Binary Indicator (No scaling needed)
    data["RAIN_ind"] = (rain_raw > 0).astype(np.float32)
    
    # Step 1b: Intensity Transformation (Log magnitude if > 0, else 0)
    rain_int = np.where(
        rain_raw > 0,                       # condition
        np.log(np.maximum(rain_raw, 1e-5)), # true case: log-transform with a small epsilon to avoid log(0)
        0.0                                 # false case: set to 0 for no rain
    ).reshape(-1, 1)
    data["RAIN_int"] = rain_int  
    data.drop(columns=["RAIN"], inplace=True)

    data = preprocess_time_features(data)

    # 6d. Wind Direction ('wd') -> Map to Angles -> Sine/Cosine
    if "wd" in data.columns:
        degrees = data["wd"].map(WD_ANGLES).fillna(0.0).values
        wd_rad = np.radians(degrees)
        data["wd_sin"] = np.sin(wd_rad)
        data["wd_cos"] = np.cos(wd_rad)
        data.drop(columns=["wd"], inplace=True)

    # # -------------------------------------------------------------------------
    # # Group 7: Categorical Station Identifier (Label Encoding for Embedding)
    # # -------------------------------------------------------------------------
    # if "station" in data.columns:
    #     if is_train:
    #         scalers["encoder_station"] = LabelEncoder()
    #         data["station"] = scalers["encoder_station"].fit_transform(data["station"].astype(str))
    #     else:
    #         data["station"] = scalers["encoder_station"].transform(data["station"].astype(str))

    data.drop(columns=["station"], inplace=True)

    # -------------------------------------------------------------------------
    # Cleanup: Drop Row Index & Year (or keep year if long-term trend modeling)
    # -------------------------------------------------------------------------
    cols_to_drop = [c for c in ["No", "year"] if c in data.columns]
    data.drop(columns=cols_to_drop, inplace=True)

    num_cols = data.select_dtypes(include=['float']).columns
    data[num_cols] = data[num_cols].astype(np.float32)

    return data

def altieri_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # -------------------------------------------------------------------------
    # 0. Handle Missing Values (Imputation)
    # -------------------------------------------------------------------------
    # Interpolate numeric features forward/backward to remove NaNs cleanly
    num_cols = data.select_dtypes(include=['float']).columns
    data = data[num_cols]
    data = data.fillna(data.mean())

    return data

@dvc_stage(root_dir=ROOT_DIR)
def main(ctx: StageContextManager):
    cfg = ctx.cfg
    paths_cfg = cfg.paths
    bma_dir_path = Path(paths_cfg.bma_dir_path)
    bma_out_path = Path(paths_cfg.dataset_path)

    final_df = []
    sorted_csv_files = sorted(bma_dir_path.glob("*.csv"), key=lambda x: x.name)
    for i, csv_file in enumerate(sorted_csv_files):
        # print(f"Processing file {i}/{len(sorted_csv_files)}: {csv_file.name}")
        df = pd.read_csv(csv_file)
        final_df.append(df)
    
    n_nodes = len(final_df)

    final_df = pd.concat(final_df, axis=0)
    preprocessed_df = gemini_preprocess(final_df)
    # for col in preprocessed_df.columns:
    #     print(f"Min and Max of {col}: {preprocessed_df[col].min()} - {preprocessed_df[col].max()}")

    tensor_data = preprocessed_df.to_numpy(dtype=np.float32)

    # reshape it to (S, N, F) where 
    # S is the number of sequences,
    # N is the number of nodes, and 
    # F is the number of features
    num_seqs = tensor_data.shape[0] // n_nodes
    tensor_data = tensor_data.reshape(num_seqs, n_nodes, tensor_data.shape[1])
    # print(f"Final tensor shape: {tensor_data.shape}")

    os.makedirs(bma_out_path.parent, exist_ok=True)
    np.savez_compressed(bma_out_path, data=tensor_data)

if __name__ == "__main__":
    main()