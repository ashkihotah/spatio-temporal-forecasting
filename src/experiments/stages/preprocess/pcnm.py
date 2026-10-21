import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import Union, List

def compute_pcnm(
    df: pd.DataFrame,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    id_col: str = 'idplant',
    n_components: int = 15,
    metric: str = 'euclidean'
) -> pd.DataFrame:
    """
    Computes Principal Coordinates of Neighbor Matrices (PCNM) spatial features 
    from geographic coordinates as described in spatial-temporal modeling literature.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing plant metadata.
    lat_col : str, default 'lat'
        Name of the column containing latitude coordinates.
    lon_col : str, default 'lon'
        Name of the column containing longitude coordinates.
    id_col : str, default 'idplant'
        Name of the column containing the plant identifier to index the output.
    n_components : int, default 15
        Number of PCNM features to return.
    metric : str, default 'euclidean'
        Distance metric to use ('euclidean' for flat coordinates, 'haversine' for 
        great-circle geographic distances).

    Returns:
    --------
    pd.DataFrame
        A DataFrame indexed by id_col containing the computed PCNM features 
        named 'pcnm1', 'pcnm2', etc.
    """
    # 1. Extract coordinates and compute the distance matrix D
    coords = df[[lat_col, lon_col]].values
    
    if metric == 'haversine':
        # Convert degrees to radians for Haversine metric
        coords_rad = np.radians(coords)
        # pdist with haversine returns distance in radians. Multiply by Earth radius (6371 km)
        D = squareform(pdist(coords_rad, metric='haversine')) * 6371.0
    else:
        # Standard Euclidean distance in coordinate degrees
        D = squareform(pdist(coords, metric='euclidean'))
    
    n = len(df)
    if n <= 1:
        raise ValueError("PCNM requires at least 2 distinct plant locations.")

    # # 2. Find the MST threshold 't'
    # # The threshold 't' is the maximum edge weight in the Minimum Spanning Tree
    # mst = minimum_spanning_tree(D)
    # t = mst.toarray().max()

    '''
    the value of t used in D is defined as the maximum distance between two PV plants
    t = max_{i, j} D[i, j]
    in order to guarantee that the data are all connected.
    '''
    t = D.max()

    # 3. Construct the truncated distance matrix D*
    # Values <= t are kept; values > t are replaced with 4 * t
    D_star = np.where(D <= t, D, 4.0 * t)
    np.fill_diagonal(D_star, 0.0)  # Ensure diagonal elements are strictly zero

    # 4. Double-centering of squared truncated distance matrix
    D_star_sq = D_star ** 2
    one_matrix = np.ones((n, n))
    H = np.eye(n) - one_matrix / n
    Delta = -0.5 * (H @ D_star_sq @ H)

    # 5. Eigen-decomposition of the centered matrix
    eigenvalues, eigenvectors = np.linalg.eigh(Delta)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Keep only strictly positive eigenvalues (ignoring numerical noise)
    tolerance = 0
    positive_mask = eigenvalues >= tolerance
    pos_eigenvalues = eigenvalues[positive_mask]
    pos_eigenvectors = eigenvectors[:, positive_mask]

    # 6. Scale eigenvectors to the length of the square root of their eigenvalues
    pcnm_matrix = pos_eigenvectors * np.sqrt(pos_eigenvalues)

    # Select requested number of components
    total_valid_components = pcnm_matrix.shape[1]
    actual_components = min(n_components, total_valid_components)
    selected_pcnm = pcnm_matrix[:, :actual_components]

    # Create column names
    col_names = [f"pcnm{i+1}" for i in range(actual_components)]
    
    # Structure output DataFrame
    output_df = pd.DataFrame(selected_pcnm, columns=col_names, index=df[id_col])
    
    return output_df

def compute_pcnm_stable(
    df: pd.DataFrame,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    id_col: str = 'idplant',
    n_components: int = 15,
    metric: str = 'euclidean'
) -> pd.DataFrame:
    """
    Computes PCNM features with enhanced numerical stability to minimize 
    floating-point propagation errors.
    """
    n = len(df)
    if n <= 1:
        raise ValueError("PCNM requires at least 2 distinct plant locations.")

    # 1. Cast coordinates immediately to float64 to prevent precision loss
    coords = df[[lat_col, lon_col]].values.astype(np.float64)
    
    if metric == 'haversine':
        coords_rad = np.radians(coords)
        # Using 6371.0088 km (IUGG mean radius) in double precision
        D = squareform(pdist(coords_rad, metric='haversine')) * 6371.0088
    else:
        D = squareform(pdist(coords, metric='euclidean'))
        
    # Enforce absolute zero on the diagonal to eliminate tiny float self-distances
    np.fill_diagonal(D, 0.0)

    # # 2. Extract MST threshold 't' in double precision
    # mst = minimum_spanning_tree(D)
    # t = np.float64(mst.toarray().max())

    '''
    the value of t used in D is defined as the maximum distance between two PV plants
    t = max_{i, j} D[i, j]
    in order to guarantee that the data are all connected.
    '''
    t = D.max()

    # 3. Construct truncated distance matrix D*
    # Float comparisons are safer with explicit boundaries
    D_star = np.where(D <= t, D, 4.0 * t)
    np.fill_diagonal(D_star, 0.0)

    # 4. Numerically stable double-centering (O(n^2) instead of O(n^3))
    # This directly avoids error propagation caused by matrix dot products.
    D_star_sq = D_star ** 2
    
    row_means = np.mean(D_star_sq, axis=1, keepdims=True)
    col_means = np.mean(D_star_sq, axis=0, keepdims=True)
    grand_mean = np.mean(D_star_sq)
    
    # Delta = -0.5 * (D_ij^2 - row_mean_i - col_mean_j + grand_mean)
    Delta = -0.5 * (D_star_sq - row_means - col_means + grand_mean)

    # Enforce strict matrix symmetry to correct any asymmetric float artifacts
    Delta = 0.5 * (Delta + Delta.T)

    # 5. Solve the symmetric eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eigh(Delta)

    # Sort in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # 6. Apply dynamic LAPACK tolerance thresholding
    # tol = n * lambda_max * machine_epsilon
    eps = np.finfo(Delta.dtype).eps
    tol = n * eigenvalues[0] * eps
    
    positive_mask = eigenvalues > tol
    pos_eigenvalues = eigenvalues[positive_mask]
    pos_eigenvectors = eigenvectors[:, positive_mask]

    # 7. Scale eigenvectors to get Principal Coordinates
    pcnm_matrix = pos_eigenvectors * np.sqrt(pos_eigenvalues)

    # Match output dimension constraints
    total_valid_components = pcnm_matrix.shape[1]
    actual_components = min(n_components, total_valid_components)
    selected_pcnm = pcnm_matrix[:, :actual_components]

    # Construct clean output dataframe
    col_names = [f"pcnm{i+1}" for i in range(actual_components)]
    output_df = pd.DataFrame(selected_pcnm, columns=col_names, index=df[id_col])
    
    return output_df

# Example Usage
if __name__ == "__main__":
    df = pd.read_parquet("dvc-stages-outputs/pv-italy/raw/plants.parquet")
    # remove all rows having NaN values
    df = df.dropna()

    # Compute PCNM using the Euclidean (geographic degrees) option
    pcnm_features = compute_pcnm(df, n_components=15, metric='euclidean')
    # MinMax normalization of the PCNM features
    pcnm_features = (pcnm_features - pcnm_features.min()) / (pcnm_features.max() - pcnm_features.min())
    
    print("Computed PCNM Features:")
    print(pcnm_features)