import pandas as pd
import numpy as np
import os
from geopy.distance import geodesic
from sklearn.metrics.pairwise import haversine_distances

def compute_distances_geopy(df):
    """
    Calculate the distance matrix using geopy's geodesic distance
    """
    n = len(df)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            coord1 = (df.loc[i, 'latitude'], df.loc[i, 'longitude'])
            coord2 = (df.loc[j, 'latitude'], df.loc[j, 'longitude'])
            dist_matrix[i, j] = geodesic(coord1, coord2).kilometers
    return dist_matrix

def compute_distances_sklearn(df):
    """
    Calculate the distance matrix using scikit-learn's haversine_distances
    """
    # sklearn haversine expects coordinates in radians
    coords_rad = np.radians(df[['latitude', 'longitude']].values)
    
    # Calculate pairwise distances in radians
    dist_matrix_rad = haversine_distances(coords_rad, coords_rad)
    
    # Multiply by Earth radius in kilometers to get distance
    r = 6371.0088 
    return dist_matrix_rad * r

def main():
    input_file = './data/raw/beijing-multisite-airquality/stations_coordinates.csv'
    output_dir = './data/interim/beijing-multisite-airquality'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the coordinates
    df = pd.read_csv(input_file)
    stations = df['station'].values
    n = len(stations)
    
    # Initialize the distance matrix using one of the methods
    # You can swap this to use compute_distances_geopy(df)
    dist_matrix = compute_distances_sklearn(df)
    # dist_matrix = compute_distances_geopy(df)
            
    # Save the distance matrix
    df_dist = pd.DataFrame(dist_matrix, index=stations, columns=stations)
    dist_output_path = os.path.join(output_dir, 'distance_matrix.csv')
    df_dist.to_csv(dist_output_path)
    print(f"Saved distance matrix to {dist_output_path}")
    
    # Compute the correlation adjacency matrix
    # a(i,j) = 1 - (d(i,j) / max_dist)
    max_dist = dist_matrix.max()
    corr_matrix = 1 - (dist_matrix / max_dist)
    
    # Save the correlation adjacency matrix
    df_corr = pd.DataFrame(corr_matrix, index=stations, columns=stations)
    corr_output_path = os.path.join(output_dir, 'correlation_adjacency_matrix.csv')
    df_corr.to_csv(corr_output_path)
    print(f"Saved correlation adjacency matrix to {corr_output_path}")

if __name__ == '__main__':
    main()
