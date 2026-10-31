from pathlib import Path

import pandas as pd
import numpy as np

from dvc_mlops import StageContextManager, dvc_stage

from experiments import ROOT_DIR
from experiments.stages.preprocess import preprocess_time_features


@dvc_stage(root_dir=ROOT_DIR)
def main(ctx: StageContextManager):
    cfg = ctx.cfg
    paths_cfg = cfg.paths
    raw_dir_path = Path(paths_cfg.raw_dataset_path)
    dataset_path = Path(paths_cfg.dataset_path)

    # load npz compressed file
    data = np.load(raw_dir_path, allow_pickle=True)
    features = data["data"] # shape (B, N, F)

    n_timesteps = features.shape[0]
    n_nodes = features.shape[1]
    
    features = features.reshape(-1, features.shape[-1]) # shape (B*N, F)


    # convert it into a pandas dataframe
    df = pd.DataFrame(features)
    # add datetimes from January 1, 2005 to December 31, 2006
    dates = pd.date_range(
        start="2005-01-01", 
        periods=n_timesteps,
        freq="h"
    )

    df["timestamp"] = pd.NaT

    for i in range(n_nodes):
        df.loc[i::n_nodes, "timestamp"] = dates

    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df.drop(columns=["timestamp"], inplace=True)

    df = preprocess_time_features(df)

    np_tensor = df.to_numpy(dtype=np.float32)
    num_seqs = np_tensor.shape[0] // n_nodes
    tensor_data = np_tensor.reshape(
        num_seqs, n_nodes, np_tensor.shape[1]
    )
    print(f"Final tensor shape: {tensor_data.shape}")
    np.savez_compressed(dataset_path, data=tensor_data)

    print(df)

if __name__ == "__main__":
    main()