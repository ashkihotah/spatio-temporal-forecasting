from pathlib import Path

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# 1. Configuration (Postgres.app defaults)
DB_USER = "armandoresta" # Automatically grabs your macOS username
DB_PASSWORD = ""  # Usually blank for Postgres.app local setup
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "armandoresta"  # Replace with your database name

# 2. Create the database connection string
# If you don't use a password, omit it from the URL string
connection_url = f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# If you do have a password, use this instead:
# connection_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

path = Path("./dvc-stages-outputs/pv-italy/raw")

# conn = create_engine(connection_url)
conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
)

# azimuth_altitude.parquet
query = "SELECT idsito as site_id, altitude, azimuth, data, ora FROM azimuth_altitude;"

# local_features.parquet
query = '''
SELECT id, idsito as site_id, idmodel as model_id, idbrand as brand_id, 
data as timestamp, temperatura_ambiente as temperature, 
irradiamento as irradiance, pressure, windspeed, 
humidity, dewpoint, windbearing, cloudcover, kwh
from info_ambientali_corrette_complete
'''

# external_ambiental_features.parquet
query = '''
SELECT idplant as site_id, idhost as host_id,
tstamp2 as timestamp, 
temperature, 
pressure, windspeed, humidity, dewpoint, 
windbearing, cloudcover
from history
'''

# external_irradiance_features.parquet
query = '''
SELECT idplant as site_id, namehostirr as hostname,
tstamp as timestamp, 
g, gd, gc, a, ad, ac, td
from irradiance_hour
inner join hostirr on hostirr.idhostirr = irradiance_hour.idhostirr
'''

df = pd.read_sql(query, con=conn)

# azimuth_altitude.parquet
# df['data'] = pd.to_datetime(df['data'], format='%d/%m')
# df['ora'] = pd.to_datetime(df['ora'], format='%H:%M:%S')
# df['timestamp'] = df.apply(lambda row: pd.Timestamp.combine(row['data'].date(), row['ora'].time()), axis=1)
# df.drop(columns=['data', 'ora'], inplace=True)

# external_irradiance_features.parquet
# df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m %H:%M:%S')
# df['year'] = df['timestamp'].dt.year
# df['month'] = df['timestamp'].dt.month
# df['hour'] = df['timestamp'].dt.hour
# irr_df.drop(columns=["timestamp"], inplace=True)

output_path = path / "external_irradiance_features.parquet"
print(f"Saving {len(df)} rows to {output_path}...")
df.to_parquet(output_path, index=False)