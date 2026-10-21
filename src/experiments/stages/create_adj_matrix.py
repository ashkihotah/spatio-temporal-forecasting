import argparse
import os

import numpy as np

def create_adj(adj_path=None, sequence=None, show=False, save=False, **kwargs):
    """
    Create adjacency matrix
    -----------------------
    Parameters:
    adj_path (String, optional): if specified, the matrix will be loaded from file, from the specified path.
    The adjacency matrix loaded this way is assumed to already have self-loops (values on the diagonal should be the highest).
    sequence (Tensor, optional): if specified, the matrix will be calculated using the correlation of the first feature between nodes.
    It must be a tensor of shape [B,T,N,F].
    Exactly one between adj_path and sequence must be specified.
    The adjacency matrix is normalized but the Laplacian is not used.
    """
    assert (adj_path is None) + (
        sequence is None
    ) == 1  # One must be specified, the other must be None

    if adj_path:
        print("Loading adjacency matrix from " + adj_path + "...")
        adj = np.load(adj_path).astype(np.float32)
        if (replicate_nodes := kwargs.get("replicate_nodes", False)) > 0:
            adj = np.tile(adj, (replicate_nodes, replicate_nodes))
    elif sequence is not None:
        print(
            "Computing adjacency matrix from sequence of shape {}...".format(
                np.shape(sequence)
            )
        )
        B, T, N, F = np.shape(sequence)
        sequence = sequence[..., 0]  # [B,T,N]
        adj = np.corrcoef(
            np.reshape(sequence, (B * T, N)), rowvar=False
        )  # Variables are nodes, which are on the columns
    else:
        raise ValueError("Exactly one between adj_path and sequence must be specified.")

    D = np.zeros_like(adj)
    for row in range(len(D)):
        D[row, row] = np.sum(adj[row])  # Degree matrix (D_ii = sum(adj[i,:]))
    sqinv_D = np.sqrt(np.linalg.inv(D))  # Calcola l'inversa e la splitta in due radici
    adj = np.matmul(sqinv_D, np.matmul(adj, sqinv_D))

    if show:
        plt.figure(figsize=(12, 12))
        sns.heatmap(adj)
    if save:
        plt.savefig(
            os.path.join(
                self.path,
                "heatmaps",
                "corr-" + datetime.datetime.now().strftime("%H%M%S") + ".png",
            )
        )
    print("Adjacency matrix loaded with shape " + str(adj.shape) + ".")

    # Checking for NaN or infinite values
    if np.isnan(adj).any():
        adj = np.nan_to_num(adj, nan=0)
        print("Adjacency matrix contains NaN values. They have been replaced with 0.")
    if np.isnan(adj).any() or np.isinf(adj).any():
        print(
            "Adjacency matrix is nan or infinite:\n"
            + str(adj)
            + "\nTerminating due to critical error."
        )
        sys.exit(1)

    return adj

