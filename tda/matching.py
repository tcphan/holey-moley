import numpy as np
import polars as pl
from collections import defaultdict
from itertools import product
from sklearn.cluster import DBSCAN


def mapper_bin_and_cluster(
    df_features: pl.DataFrame,
    df_umap: pl.DataFrame,
    n_bins: int = 5,
    overlap: float = 0.2,
    eps: float = 0.5,
    min_samples: int = 3,
) -> pl.DataFrame:
    """
    Bins UMAP projection into overlapping intervals and clusters points inside each interval.

    Parameters:
    -----------
    df_features : pl.DataFrame
        The high-dimensional features to cluster.
    df_umap : pl.DataFrame
        The UMAP projection of the data.
    n_bins : int
        The number of bins to divide the UMAP projection into.
    overlap : float
        The amount of overlap between bins.
    eps : float
        The epsilon value for DBSCAN (density-based spatial clustering algorithm).
    min_samples : int
        The minimum number of samples for DBSCAN.

    Returns:
    --------
    pl.DataFrame
        A DataFrame containing the cluster memberships.
    """

    umap_matrix = df_umap.to_numpy()
    n_samples, n_components = umap_matrix.shape

    # 1. Compute bin boundaries and widths
    mins = umap_matrix.min(axis=0)
    maxs = umap_matrix.max(axis=0)
    steps = (maxs - mins) / n_bins
    margins = steps * overlap

    # 2. Map every point to its base bin indices (0 to n_bins-1)
    base_bins = np.clip(((umap_matrix - mins) / steps).astype(int), 0, n_bins - 1)

    # 3. Build a spatial index: map active bin coordinate tuples to point indices
    # Because bins overlap, a point in base bin `b` can also belong to neighboring bins
    active_hypercubes = defaultdict(list)

    for i in range(n_samples):
        # Determine all hypercubes this point falls into due to overlap
        point_bin_ranges = []
        for d in range(n_components):
            val = umap_matrix[i, d]
            # Find all bin indices for dimension `d` whose interval covers `val`
            matching_bins = [
                b
                for b in range(n_bins)
                if (mins[d] + b * steps[d] - margins[d])
                <= val
                <= (mins[d] + (b + 1) * steps[d] + margins[d])
            ]
            point_bin_ranges.append(matching_bins)

        # Add point index to all intersecting active hypercube keys
        for cube_key in product(*point_bin_ranges):
            active_hypercubes[cube_key].append(i)

    # 4. Only execute DBSCAN on hypercubes that meet min_samples
    cluster_nodes = []
    features_matrix = df_features.to_numpy()

    hypercube_num = 1
    total_hypercubes = len(active_hypercubes)
    for cube_key, indices in active_hypercubes.items():

        if len(indices) < min_samples:
            print(
                f"Skipping grid cell {hypercube_num} / {total_hypercubes} due to insufficient data points."
            )
            continue

        bin_data = features_matrix[indices]
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(bin_data)

        bin_id_str = "bin_" + "_".join(map(str, cube_key))
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            node_members = np.array(indices)[labels == cluster_id]

            bin_dict = {
                "bin_id": bin_id_str,
                "cluster_id": cluster_id,
                "point_indices": node_members.tolist(),
                "node_size": len(node_members),
            }
            cluster_nodes.append(bin_dict)

        n_clusters = len(set(labels) - {-1})
        print(
            f"Found {n_clusters} clusters in grid cell {hypercube_num} / {total_hypercubes}."
        )
        hypercube_num += 1

    return pl.DataFrame(cluster_nodes)
