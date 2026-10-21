import numpy as np
import pandas as pd

def preprocess_time_features(data: pd.DataFrame) -> pd.DataFrame:
    # 6a. Hour (0-23)
    hour_rad = 2 * np.pi * data["hour"].values / 24.0
    data["hour_sin"] = np.sin(hour_rad)
    data["hour_cos"] = np.cos(hour_rad)

    # 6b. Month (1-12)
    month_rad = 2 * np.pi * (data["month"].values - 1) / 12.0
    data["month_sin"] = np.sin(month_rad)
    data["month_cos"] = np.cos(month_rad)

    # 6c. Day (1-31)
    day_rad = 2 * np.pi * (data["day"].values - 1) / 31.0
    data["day_sin"] = np.sin(day_rad)
    data["day_cos"] = np.cos(day_rad)

    data.drop(columns=["hour", "month", "day"], inplace=True)
    return data