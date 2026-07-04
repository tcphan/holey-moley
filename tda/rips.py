import numpy as np
import polars as pl


class VietorisRips:
    """
    Constructs the Vietoris-Rips filtration of a point cloud or distance matrix.
    """

    def __init__(self, max_dim=2, max_epsilon=float("inf")):
        """
        Initialize the Vietoris-Rips complex builder.

        Parameters:
        -----------
        max_dim : int
            The maximum dimension of simplices to construct (e.g., 0 for vertices,
            1 for edges, 2 for triangles).
        max_epsilon : float
            The maximum filtration value (threshold distance) for adding simplices.
        """
        self.max_dim = max_dim
        self.max_epsilon = max_epsilon
        self.filtration = None
        self.persistence_pairs = None

    def fit_transform(self, df: pl.DataFrame) -> list:
        """
        Computes the Vietoris-Rips filtration for the input data.

        Parameters:
        -----------
        df : pl.DataFrame
            Dataframe of input features.

        Returns:
        --------
        filtration : list of tuples
            A sorted list of simplices forming the filtration.
            Each element is a tuple: (simplex_tuple, birth_time)
            where simplex_tuple is a tuple of vertex indices (sorted) and birth_time is a float.
        """
        n_samples = df.height
        df_with_id = (
            df.with_row_index("id").with_columns([pl.col("id").cast(pl.Int64)]).lazy()
        )

        # Compute pairwise Euclidean distance
        # To optimize, filter first where a.id < b.id to avoid self-distance
        pairs_df = df_with_id.join(df_with_id, how="cross", suffix="_right").filter(
            pl.col("id") < pl.col("id_right")
        )
        feature_cols = [c for c in df.columns]
        distance_expr = pl.sum_horizontal(
            [(pl.col(c) - pl.col(f"{c}_right")) ** 2 for c in feature_cols]
        ).sqrt()

        # Filter edges by max_epsilon
        edges_base = (
            pairs_df.with_columns(distance_expr.alias("distance"))
            .filter(pl.col("distance") <= self.max_epsilon)
            .select(["id", "id_right", "distance"])
            .collect(streaming=True)
        )

        # Initialize simplices list by dimension
        simplices_by_dim = {}

        # Dimension 0: Vertices (all birth times are 0.0)
        simplices_by_dim[0] = pl.DataFrame(
            {"v_0": list(range(n_samples)), "birth_time": 0.0, "dim": 0},
            schema={"v_0": pl.Int64, "birth_time": pl.Float64, "dim": pl.Int32},
        )

        # Dimension 1: Edges
        simplices_by_dim[1] = edges_base.select(
            [
                pl.col("id").cast(pl.Int64).alias("v_0"),
                pl.col("id_right").cast(pl.Int64).alias("v_1"),
                pl.col("distance").alias("birth_time"),
                pl.lit(1, dtype=pl.Int32).alias("dim"),
            ]
        )

        # Inductive step: generate higher-dimensional simplices
        for d in range(2, self.max_dim + 1):

            # To find a d-simplex, we join a (d-1)-simplex with edges
            # We unnest the fields to match lower dimensional vertices with edges
            prev_df = simplices_by_dim[d - 1]
            last_vertex_col = f"v_{d-1}"
            new_vertex_col = f"v_{d}"

            # We take the last node of the current face, and look for edges connecting it to a higher node 'v'
            joined = prev_df.join(
                edges_base.select(
                    [
                        pl.col("id"),
                        pl.col("id_right").alias(new_vertex_col),
                        pl.col("distance"),
                    ]
                ),
                left_on=last_vertex_col,
                right_on="id",
            )

            if joined.is_empty():
                simplices_by_dim[d] = pl.DataFrame(schema=simplices_by_dim[1].schema)
                continue

            # To ensure it is a valid clique, the new vertex must also connect to the FIRST vertex (v_0)
            valid_simplices = joined.join(
                edges_base.select(
                    [
                        pl.col("id").cast(pl.Int64),
                        pl.col("id_right").cast(pl.Int64),
                        pl.col("distance").alias("closing_distance"),
                    ]
                ),
                left_on=["v_0", new_vertex_col],
                right_on=["id", "id_right"],
            )

            # Calculate birth time as the max distance among components
            birth_expr = pl.max_horizontal(
                ["birth_time", "distance", "closing_distance"]
            )

            vertex_cols = [f"v_{i}" for i in range(d + 1)]
            simplices_by_dim[d] = valid_simplices.select(
                [
                    *vertex_cols,
                    birth_expr.alias("birth_time"),
                    pl.lit(d, dtype=pl.Int32).alias("dim"),
                ]
            )

        # Consolidate and convert back to the format required by the boundary matrix
        all_dfs = [
            simplices_by_dim[d]
            for d in range(self.max_dim + 1)
            if d in simplices_by_dim and not simplices_by_dim[d].is_empty()
        ]
        all_simplices_df = (
            pl.concat(all_dfs, how="diagonal")
            .lazy()
            .sort(["birth_time", "dim"])
            .collect(streaming=True)
        )

        self.filtration = []
        vertex_cols = [c for c in all_simplices_df.columns if c.startswith("v_")]
        for row in all_simplices_df.select([*vertex_cols, "birth_time"]).to_dicts():
            # Gather all non-null vertex indices into a sorted tuple
            simplex_tuple = tuple(
                sorted(int(row[v]) for v in vertex_cols if row[v] is not None)
            )
            self.filtration.append((simplex_tuple, row["birth_time"]))

        return self.filtration

    def get_simplices_for_dim(self, dim: int) -> list:
        """
        Retrieves the simplices for the specified i-th dimension.

        Parameters:
        -----------
        dim : int
            The dimension of simplices to retrieve.

        Returns:
        --------
        list of tuples
            A list of simplices for the specified dimension.
            Each element is a tuple: (simplex_tuple, birth_time)
            where simplex_tuple is a tuple of vertex indices (sorted) and birth_time is a float.
        """

        if self.filtration is None:
            raise ValueError(
                "Filtration has not been computed. Call fit_transform() first."
            )

        return [s for s in self.filtration if len(s[0]) == dim + 1]

    def __create_boundary_matrix(self, simplex_to_idx):
        """
        Internal helper to construct the boundary matrix for the Rips filtration.

        Parameters:
        -----------
        simplex_to_idx : dict
            A dictionary mapping each simplex (tuple of vertex indices) to its column index in the matrix.

        Returns:
        --------
        boundaries : list
            A list of numpy arrays, where each array contains the row indices of the faces
            (lower-dimensional simplices) of the simplex at the corresponding column index.
        """

        simplices = list(simplex_to_idx.keys())
        boundaries = []
        for col_idx in range(len(simplices)):
            simplex = simplices[col_idx]
            dim = len(simplex) - 1
            if dim == 0:
                boundaries.append(np.array([], dtype=np.int32))
            else:
                faces = []
                for i in range(len(simplex)):
                    face = simplex[:i] + simplex[i + 1 :]
                    idx = simplex_to_idx.get(face, -1)
                    if idx != -1:
                        faces.append(idx)
                faces.sort(reverse=True)
                boundaries.append(np.array(faces, dtype=np.int32))
        return boundaries

    def compute_birth_death_pairs(self):
        """
        Computes birth-death pairs from the filtration.

        Returns:
        --------
        persistence_pairs : dict
            A dictionary mapping dimension -> list of (birth_time, death_time) tuples.
        """

        num_simplices = len(self.filtration)
        simplex_to_idx = {simplex: i for i, (simplex, _) in enumerate(self.filtration)}
        self.persistence_pairs = {}

        # pivot_to_col[r] stores the column index c that has its pivot at row r (-1 means unassigned)
        pivot_to_col = np.full(num_simplices, -1, dtype=np.int32)
        # is_cycle represents if a simplex is a cycle (has no pivot)
        is_cycle = np.ones(num_simplices, dtype=np.bool_)

        # Pre-extract all simplex metadata into fast numpy arrays for O(1) loop checks
        birth_times = np.array([item[1] for item in self.filtration], dtype=np.float64)
        dimensions = np.array(
            [len(item[0]) - 1 for item in self.filtration], dtype=np.int32
        )

        # Calculate the boundary matrix
        boundaries = self.__create_boundary_matrix(simplex_to_idx)

        # Core boundary matrix reduction Loop
        for col_idx in range(num_simplices):
            dim = dimensions[col_idx]
            if dim == 0:
                continue

            # Local working copy of the boundary for this column
            boundary_indices = list(boundaries[col_idx])

            while boundary_indices:
                pivot_row = boundary_indices[0]
                other_col = pivot_to_col[pivot_row]

                if other_col != -1:
                    # Vectorized-style XOR operation using Python sets (highly optimized in C)
                    boundary_indices = list(
                        set(boundary_indices) ^ set(boundaries[other_col])
                    )
                    boundary_indices.sort(reverse=True)
                else:
                    # pivot_row is born, col_idx kills it
                    pivot_to_col[pivot_row] = col_idx
                    # col_idx is now a boundary, not a cycle
                    is_cycle[col_idx] = False

                    birth_dim = dimensions[pivot_row]
                    b_time = birth_times[pivot_row]
                    d_time = birth_times[
                        col_idx
                    ]  # The current simplex's birth time is the death time

                    if birth_dim not in self.persistence_pairs:
                        self.persistence_pairs[birth_dim] = []

                    if d_time > b_time:
                        self.persistence_pairs[birth_dim].append((b_time, d_time))
                    break

        # Collect essential features (unpaired cycles that live to infinity)
        for col_idx in range(num_simplices):
            if is_cycle[col_idx] and (pivot_to_col[col_idx] == -1):
                dim = dimensions[col_idx]
                if dim not in self.persistence_pairs:
                    self.persistence_pairs[dim] = []
                self.persistence_pairs[dim].append((birth_times[col_idx], float("inf")))

        return self.persistence_pairs
