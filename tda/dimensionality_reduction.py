import numpy as np
import polars as pl
from numba import njit
from pynndescent import NNDescent


@njit(fastmath=True)
def _optimize_layout_numba(
    embedding, head, tail, weights, n_epochs, n_samples, n_components, lr, a, b
):
    n_edges = head.shape[0]

    for epoch in range(n_epochs):
        alpha = lr * (1.0 - epoch / n_epochs)

        for e in range(n_edges):
            i = head[e]
            j = tail[e]
            w = weights[e]

            # Vectorized gradient distance
            dist_sq = 0.0
            for d in range(n_components):
                diff = embedding[i, d] - embedding[j, d]
                dist_sq += diff * diff

            dist_sq = max(dist_sq, 1e-6)

            # Attractive Force
            attr_coeff = -(2.0 * a * b * (dist_sq ** (b - 1.0))) / (
                1.0 + a * (dist_sq**b)
            )

            for d in range(n_components):
                diff = embedding[i, d] - embedding[j, d]
                grad = attr_coeff * diff * w
                embedding[i, d] -= alpha * grad
                embedding[j, d] += alpha * grad

            # Repulsive Force (2 Negative Samples per edge)
            for _ in range(2):
                neg_j = np.random.randint(0, n_samples)
                if neg_j == i:
                    continue

                neg_dist_sq = 0.0
                for d in range(n_components):
                    diff = embedding[i, d] - embedding[neg_j, d]
                    neg_dist_sq += diff * diff

                neg_dist_sq = max(neg_dist_sq, 1e-6)
                rep_coeff = (2.0 * b) / (
                    (0.001 + neg_dist_sq) * (1.0 + a * (neg_dist_sq**b))
                )

                for d in range(n_components):
                    diff = embedding[i, d] - embedding[neg_j, d]
                    embedding[i, d] += alpha * (rep_coeff * diff * (1.0 - w)) * 0.1

    return embedding


class UMAP:
    """
    A class to perform dimensionality reduction using UMAP algorithm.
    """

    def __init__(self, n_neighbors=15, n_components=2, n_epochs=200, lr=1.0):
        """
        Initializes the UMAP class.

        Parameters:
        -----------
        n_neighbors : int
            The number of neighbors to consider for each point.
        n_components : int
            The number of dimensions to reduce the data to.
        n_epochs : int
            The number of epochs to train the model.
        lr : float
            The learning rate for the optimization process.
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.lr = lr

    def _build_fuzzy_simplicial_set(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Construct high-dimensional topological fuzzy simplicial set using Polars.

        Parameters:
        -----------
        X : pl.DataFrame
            The data to reduce the dimensionality of.

        Returns:
        --------
        pl.DataFrame
            The high-dimensional topological fuzzy simplicial set.
        """
        n_samples = X.shape[0]
        X_np = X.to_numpy()

        # 1.Extract k-NN graph for each node
        index = NNDescent(X_np, n_neighbors=self.n_neighbors, metric="euclidean")
        index.prepare()
        knn_indices, knn_dists = index.neighbor_graph

        # 2. Extract local connectivity threshold rho (distance to 2nd nearest neighbor, ignoring self)
        # knn_dists[:, 0] is distance to self (0.0)
        rho = knn_dists[:, 1]

        # 3. Vectorized binary search for sigma_i per node
        target = np.log2(self.n_neighbors)
        lows = np.full(n_samples, 1e-3)
        highs = np.full(n_samples, 100.0)
        sigmas = np.ones(n_samples)

        dists_minus_rho = np.maximum(0.0, knn_dists - rho[:, None])

        for _ in range(20):
            mids = (lows + highs) / 2.0
            vals = np.sum(np.exp(-dists_minus_rho / mids[:, None]), axis=1)

            mask = vals > target
            highs[mask] = mids[mask]
            lows[~mask] = mids[~mask]
            sigmas = mids

        # 4. Directional fuzzy membership strength: mu_{i->j}
        mus = np.exp(-dists_minus_rho / sigmas[:, None])

        # 5. Build edge list using Polars without cross-joining
        i_indices = np.repeat(np.arange(n_samples), self.n_neighbors)
        j_indices = knn_indices.ravel()
        mu_weights = mus.ravel()

        knn_full = pl.DataFrame({"i": i_indices, "j": j_indices, "mu": mu_weights})

        # 6. Symmetrize edge weights (Fuzzy Set Union: w = a + b - a*b)
        reversed_edges = knn_full.select(
            [
                pl.col("i").alias("j"),
                pl.col("j").alias("i"),
                pl.col("mu").alias("mu_ba"),
            ]
        )

        fuzzy_graph = (
            knn_full.join(reversed_edges, on=["i", "j"], how="outer")
            .with_columns(
                [
                    pl.col("mu").fill_null(0.0),
                    pl.col("mu_ba").fill_null(0.0),
                ]
            )
            .with_columns(
                weight=pl.col("mu") + pl.col("mu_ba") - (pl.col("mu") * pl.col("mu_ba"))
            )
            .filter((pl.col("i") != pl.col("j")) & (pl.col("weight") > 1e-4))
        )

        return fuzzy_graph.select(["i", "j", "weight"])

    def fit_transform(self, X: pl.DataFrame, random_state: int = 42) -> pl.DataFrame:
        """
        Perform dimensionality reduction using UMAP algorithm.

        Parameters:
        -----------
        X : pl.DataFrame
            The data to reduce the dimensionality of.
        random_state : int, optional
            The random state to use for the optimization process.

        Returns:
        ---------
        pl.DataFrame
            The low-dimensional representation of the data.
        """
        n_samples = X.shape[0]

        # 1. Extract fuzzy graph
        fuzzy_edges = self._build_fuzzy_simplicial_set(X)

        head = fuzzy_edges["i"].to_numpy().astype(np.int32)
        tail = fuzzy_edges["j"].to_numpy().astype(np.int32)
        weights = fuzzy_edges["weight"].to_numpy().astype(np.float32)

        # 2. Initialize low-dimensional coordinates
        rng = np.random.default_rng(random_state)
        embedding = rng.normal(scale=10.0, size=(n_samples, self.n_components))

        # 3. Optimize Layout via Fuzzy Cross-Entropy SGD
        embedding = _optimize_layout_numba(
            embedding=embedding,
            head=head,
            tail=tail,
            weights=weights,
            n_epochs=self.n_epochs,
            n_samples=n_samples,
            n_components=self.n_components,
            lr=self.lr,
            a=1.0,
            b=1.0,
        )

        schema = [f"umap_{d + 1}" for d in range(self.n_components)]
        return pl.DataFrame(embedding, schema=schema)
