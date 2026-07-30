import numpy as np
import polars as pl
import matplotlib.pyplot as plt


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

        # 1. Pairwise Cartesian Product in Polars
        df_x = pl.DataFrame({"i": range(n_samples), "vec_i": X.rows()})
        df_y = pl.DataFrame({"j": range(n_samples), "vec_j": X.rows()})

        pairs = df_x.join(df_y, how="cross")

        # 2. Compute Euclidean distances across all pairs
        pairs = pairs.with_columns(
            dist=pl.struct(["vec_i", "vec_j"]).map_elements(
                lambda s: float(
                    np.linalg.norm(np.array(s["vec_i"]) - np.array(s["vec_j"]))
                ),
                return_dtype=pl.Float64,
            )
        )

        # 3. Extract k-NN graph for each node
        knn = (
            pairs.sort(["i", "dist"])
            .group_by("i", maintain_order=True)
            .head(self.n_neighbors)
        )

        # 4. Find local connectivity threshold rho_i (distance to nearest neighbor > 0)
        rho_df = (
            knn.filter(pl.col("dist") > 0)
            .group_by("i")
            .agg(pl.col("dist").min().alias("rho"))
        )
        knn_with_rho = knn.join(rho_df, on="i", how="left").with_columns(
            pl.col("rho").fill_null(0.0)
        )

        # 5. Solve for sigma_i per node via binary search
        target = np.log2(self.n_neighbors)
        sigmas = []

        for i in range(n_samples):
            sub = knn_with_rho.filter(pl.col("i") == i)
            dists = sub["dist"].to_numpy()
            rho = sub["rho"][0]

            low, high = 1e-3, 100.0
            sigma = 1.0
            for _ in range(20):
                mid = (low + high) / 2.0
                val = np.sum(np.exp(-np.maximum(0.0, dists - rho) / mid))
                if val > target:
                    high = mid
                else:
                    low = mid
                sigma = mid
            sigmas.append(sigma)

        sigma_df = pl.DataFrame({"i": np.arange(n_samples), "sigma": sigmas})
        knn_full = knn_with_rho.join(sigma_df, on="i", how="left")

        # 6. Directional fuzzy membership strength: mu_{i->j}
        knn_full = knn_full.with_columns(
            mu=np.exp(
                -np.maximum(0.0, pl.col("dist") - pl.col("rho")) / pl.col("sigma")
            )
        )

        # 7. Symmetrize edge weights (Fuzzy Set Union: w = a + b - a*b)
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

        # Step 1: Extract Topological Graph
        fuzzy_edges = self._build_fuzzy_simplicial_set(X)
        edges = fuzzy_edges.to_numpy()  # Columns: [i, j, weight]

        # Step 2: Initialize low-dimensional coordinates
        rng = np.random.default_rng(random_state)
        embedding = rng.normal(scale=10.0, size=(n_samples, self.n_components))

        # Step 3: Optimize Layout via Fuzzy Cross-Entropy SGD
        a, b = 1.0, 1.0  # Membership curve parameters: 1 / (1 + a * d^(2b))

        for epoch in range(self.n_epochs):
            alpha = self.lr * (1.0 - epoch / self.n_epochs)

            for i_idx, j_idx, w in edges:
                i, j = int(i_idx), int(j_idx)

                diff = embedding[i] - embedding[j]
                dist_sq = np.dot(diff, diff) + 1e-6

                # Attractive Force (pull connected simplicial nodes closer)
                attr_coeff = -(2.0 * a * b * (dist_sq ** (b - 1.0))) / (
                    1.0 + a * (dist_sq**b)
                )
                grad = attr_coeff * diff * w

                embedding[i] -= alpha * grad
                embedding[j] += alpha * grad

                # Repulsive Force (Negative Sampling)
                for _ in range(2):
                    neg_j = rng.integers(0, n_samples)
                    if neg_j == i:
                        continue
                    neg_diff = embedding[i] - embedding[neg_j]
                    neg_dist_sq = np.dot(neg_diff, neg_diff) + 1e-6

                    rep_coeff = (2.0 * b) / (
                        (0.001 + neg_dist_sq) * (1.0 + a * (neg_dist_sq**b))
                    )
                    embedding[i] += alpha * (rep_coeff * neg_diff * (1.0 - w)) * 0.1

        schema = [f"umap_{d + 1}" for d in range(self.n_components)]
        return pl.DataFrame(embedding, schema=schema)
