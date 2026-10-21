from pathlib import Path

import numpy as np
import pandas as pd

from dvc_mlops import StageContextManager, dvc_stage

from experiments import ROOT_DIR
from experiments.stages.preprocess import preprocess_time_features

import numpy as np
import pandas as pd

def generate_missing_timestamp_rows(df: pd.DataFrame) -> pd.DataFrame:
    reindexed_slices = []

    # Be careful: Here we are assuming that each site has the same
    # min-max date range. We removed site 8 for this reason!
    # (since site 8 has a different operational timeframe)

    # Process each site separately to handle different operational timeframes
    for site_id, site_data in df.groupby('site_id'):
        min_date = site_data['timestamp'].min().date()
        max_date = site_data['timestamp'].max().date()
        
        # Generate complete sequence of calendar days
        all_days = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Generate expected 19 hours (02:00 to 20:00) per day
        expected_grid = []
        for day in all_days:
            for hour in range(2, 21):  # 2 to 20 inclusive
                expected_grid.append(pd.Timestamp(day).replace(hour=hour))
                
        grid_df = pd.DataFrame({'timestamp': expected_grid})
        grid_df['site_id'] = site_id
        
        # Left merge original data onto the complete grid (missing hours become NaN)
        merged = pd.merge(grid_df, site_data, on=['site_id', 'timestamp'], how='left')
        
        reindexed_slices.append(merged)
        
    df_grid = pd.concat(reindexed_slices, ignore_index=True)
    return df_grid

def integrate_azimuth_altitude(df: pd.DataFrame, raw_dir_path: Path) -> pd.DataFrame:
    az_alt_df = pd.read_parquet(raw_dir_path / "azimuth_altitude.parquet")
    dfs = []
    for year in [2012, 2013, 2014]:
        new_rows = az_alt_df.copy()
        new_rows['timestamp'] = new_rows['timestamp'].apply(lambda x: x.replace(year=year))
        dfs.append(new_rows)
    az_alt_df = pd.concat(dfs, ignore_index=True)
    df = df.merge(az_alt_df, on=['site_id', 'timestamp'], how='left')
    return df


def impute_azimuth_altitude(df: pd.DataFrame) -> pd.Index:
    az_alt_nan_indexes = df[df["azimuth"].isnull() | df["altitude"].isnull()].index

    # impute remaining 29th February NaN value
    # in azimuth and altitude using 
    # azimuth and altitude values from 
    # the last day of february (28th) 
    # and the first day of march (1st)
    for site_id, site_data in df.groupby('site_id'):
        # Get the last day of February and the first day of March for each year
        feb_28 = site_data[(site_data['timestamp'].dt.month == 2) & (site_data['timestamp'].dt.day == 28)]
        mar_1 = site_data[(site_data['timestamp'].dt.month == 3) & (site_data['timestamp'].dt.day == 1)]
        
        # Create a mapping of timestamps to azimuth and altitude values
        az_alt_mapping = pd.concat([feb_28, mar_1]).set_index('timestamp')[['azimuth', 'altitude']].to_dict('index')
        
        # Impute NaN values for February 29th using the mapping
        for idx in site_data[(site_data['timestamp'].dt.month == 2) & (site_data['timestamp'].dt.day == 29)].index:
            timestamp = site_data.loc[idx, 'timestamp']
            year = timestamp.year
            
            # Get the corresponding February 28th and March 1st timestamps for the same year
            feb_28_timestamp = pd.Timestamp(year=year, month=2, day=28, hour=timestamp.hour)
            mar_1_timestamp = pd.Timestamp(year=year, month=3, day=1, hour=timestamp.hour)
            
            # Impute azimuth and altitude values as the average of February 28th and March 1st values
            if feb_28_timestamp in az_alt_mapping and mar_1_timestamp in az_alt_mapping:
                azimuth_avg = (az_alt_mapping[feb_28_timestamp]['azimuth'] + az_alt_mapping[mar_1_timestamp]['azimuth']) / 2
                altitude_avg = (az_alt_mapping[feb_28_timestamp]['altitude'] + az_alt_mapping[mar_1_timestamp]['altitude']) / 2
                
                df.loc[idx, 'azimuth'] = azimuth_avg
                df.loc[idx, 'altitude'] = altitude_avg

    return az_alt_nan_indexes

def impute_ambiental_features(df: pd.DataFrame, raw_dir_path: Path) -> pd.Index:
    ambiental_nan_indexes = df[df.drop(columns=["kwh", "irradiance"]).isnull().any(axis=1)].index
    rows_to_fill = df.loc[ambiental_nan_indexes][['timestamp', 'site_id']]

    ext_amb_df = pd.read_parquet(raw_dir_path / "external_ambiental_features.parquet")
    ext_amb_df = ext_amb_df[['timestamp', 'site_id', 'temperature', 'pressure', 'windspeed', 'humidity', 'dewpoint', 'windbearing', 'cloudcover']]

    fill_values = ext_amb_df.merge(rows_to_fill, on=['timestamp', 'site_id'], how='inner')

    ambiental_cols_to_fill = list(fill_values.columns)
    df.loc[ambiental_nan_indexes, ambiental_cols_to_fill] = fill_values[ambiental_cols_to_fill].values

    return ambiental_nan_indexes

def impute_irradiance(df: pd.DataFrame, raw_dir_path: Path) -> pd.Index:
    irradiance_nan_indexes = df[df['irradiance'].isnull()].index
    rows_to_fill = df.loc[irradiance_nan_indexes][['timestamp', 'site_id']]
    rows_to_fill["year"] = rows_to_fill["timestamp"].apply(lambda x: int(x.year))
    rows_to_fill["month"] = rows_to_fill["timestamp"].apply(lambda x: int(x.month))
    rows_to_fill["hour"] = rows_to_fill["timestamp"].apply(lambda x: int(x.hour))
    rows_to_fill.drop(columns=["timestamp"], inplace=True)
    rows_to_fill["index"] = rows_to_fill.index

    ext_irr_df = pd.read_parquet(raw_dir_path / "external_irradiance_features.parquet")

    fill_values = ext_irr_df.merge(rows_to_fill, on=['year', 'month', 'hour', 'site_id'], how='inner')
    indexes_to_fill = fill_values["index"]
    df.loc[indexes_to_fill, "irradiance"] = fill_values["a"].values

    df["irradiance"] = df["irradiance"].fillna(0)

    return irradiance_nan_indexes

def impute_kwh(df: pd.DataFrame) -> pd.Index:
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour

    # 2. Weather Features Fallback: Impute any remaining NaNs using Site-Month-Hour averages
    grp_month_hour = ['site_id', 'month', 'hour']

    # 3. Target Feature (kWh): Impute using Site-Month-Hour averages (as per paper Section A)
    target_climatology = df.groupby(grp_month_hour)["kwh"].transform('mean')

    kwh_nan_indexes = df[df["kwh"].isnull()].index

    df["kwh"] = df["kwh"].fillna(target_climatology)

    df.drop(columns=['year', 'month', 'hour'], inplace=True)

    return kwh_nan_indexes


def preprocess(raw_dir_path: Path, dataset_path: Path):
    local_features = pd.read_parquet(raw_dir_path / "local_features.parquet")
    altieri_mask = (local_features["timestamp"] > pd.Timestamp("2012-01-01 20:00:00")) & (local_features['site_id'] != 8)
    local_features = local_features[altieri_mask].drop(columns=["id", "model_id", "brand_id"])

    features = generate_missing_timestamp_rows(local_features)
    features.sort_values(by=["timestamp", "site_id"], inplace=True)

    features = integrate_azimuth_altitude(features, raw_dir_path)

    az_alt_nan_indexes = impute_azimuth_altitude(features)
    ambiental_nan_indexes = impute_ambiental_features(features, raw_dir_path)
    irradiance_nan_indexes = impute_irradiance(features, raw_dir_path)
    kwh_nan_indexes = impute_kwh(features)

    nan_indexes_from_feat = {
        "azimuth": az_alt_nan_indexes,
        "altitude": az_alt_nan_indexes,
        
        "irradiance": irradiance_nan_indexes,

        "kwh": kwh_nan_indexes,

        # ambiental_nan_indexes
        "temperature": ambiental_nan_indexes,
        "pressure": ambiental_nan_indexes,
        "windspeed": ambiental_nan_indexes,
        "humidity": ambiental_nan_indexes,
        "dewpoint": ambiental_nan_indexes,
        "windbearing": ambiental_nan_indexes,
        "cloudcover": ambiental_nan_indexes,
        
    }

    # Preprocess time features
    features["hour"] = features["timestamp"].dt.hour
    features["month"] = features["timestamp"].dt.month
    features["day"] = features["timestamp"].dt.day
    features = preprocess_time_features(features)

    # spatial_features = pd.read_parquet(raw_dir_path / "it_spatial.parquet")
    # features = features.merge(spatial_features, on="site_id", how="inner")

    # features.to_parquet(dataset_path.with_suffix(".parquet"))
    features.sort_values(by=["timestamp", "site_id"], inplace=True)
    features.drop(columns=["timestamp", "site_id"], inplace=True)

    kwh_col = features.pop("kwh")
    features.insert(0, "kwh", kwh_col)

    np_tensor = features.to_numpy(dtype=np.float32)
    n_nodes = 17
    num_seqs = np_tensor.shape[0] // n_nodes
    tensor_data = np_tensor.reshape(num_seqs, n_nodes, np_tensor.shape[1])
    print(f"Final tensor shape: {tensor_data.shape}")
    np.savez_compressed(dataset_path, data=tensor_data)

@dvc_stage(root_dir=ROOT_DIR)
def main(ctx: StageContextManager):
    cfg = ctx.cfg
    paths_cfg = cfg.paths
    raw_dir_path = Path(paths_cfg.raw_dir_path)
    dataset_path = Path(paths_cfg.dataset_path)

    preprocess(raw_dir_path, dataset_path)

if __name__ == "__main__":
    main()