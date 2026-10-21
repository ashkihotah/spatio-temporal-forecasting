
from collections import defaultdict
from pathlib import Path
import re

import pandas as pd

DATA_DIR_PATH = Path("dvc-stages-outputs/lightsource/raw/LIGHTSOURCE")
# feature regex capturing feature name in a group named "feature" (e.g. lightsource_1.csv -> feature=1)
FEATURE_REGEX = re.compile(r"lightsource_(?P<feature>.+)\.csv")

def main():
    
    csv_files = sorted(DATA_DIR_PATH.glob("*.csv"), key=lambda x: x.name)
    df = defaultdict(list)
    n_nodes = []
    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            file_name = csv_file.name
            feature_match = FEATURE_REGEX.match(file_name)
            if not feature_match:
                raise ValueError(f"Could not extract feature name from file name: {file_name}")

            feature_name = feature_match.group("feature")

            if feature_name in ["data_hourly", "data"]:
                continue

            feature_name = f.readline().strip()
            lines = f.readlines()
            n_nodes.append(len(lines))
            for line in lines:
                values = None

                try:
                    values = [float(x) for x in line.strip().split(",")]
                except ValueError:
                    pass

                try:
                    values = [pd.to_datetime(x) for x in line.strip().split(",")]
                except ValueError:
                    pass

                if values is None:
                    raise ValueError(f"Could not parse line: {line.strip()} in file: {csv_file.name}")
                
                df[feature_name].extend(values)

    # check if all features have the same number of nodes
    if len(set(n_nodes)) > 1:
        raise ValueError(f"Features have different number of nodes: {n_nodes}")
    n_nodes = n_nodes[0]

    # read data_hourly.csv to get the number of time steps
    data_hourly_file = DATA_DIR_PATH / "lightsource_data_hourly.csv"
    with open(data_hourly_file, "r") as f:
        lines = f.readlines()
        dates = [pd.to_datetime(line) for line in lines]
        df["timestamp"] = dates * n_nodes

    n_time_steps = len(df[list(df.keys())[0]]) // n_nodes

    df["node"] = [n for n in range(n_nodes) for _ in range(n_time_steps)]
    
    df = pd.DataFrame(df)

    df["node"] = df["node"].astype("category")
    df.to_parquet(DATA_DIR_PATH / "lightsource.parquet", index=False)

if __name__ == "__main__":
    main()