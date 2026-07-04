import numpy as np
import polars as pl


class VietorisRips:
    """
    Constructs the Vietoris-Rips filtration of a point cloud or distance matrix.
    """

    def __init__(
        self, max_dim: int = 2, max_epsilon=float("inf"), batch_size: int = 1000
    ):
        """
        Initialize the Vietoris-Rips complex builder.

        Parameters:
        -----------
        max_dim : int
            The maximum dimension of simplices to construct (e.g., 0 for vertices,
            1 for edges, 2 for triangles).
        max_epsilon : float
            The maximum filtration value (threshold distance) for adding simplices.
        batch_size : int
            The size of the batches to process.
        """
        self.max_dim = max_dim
        self.max_epsilon = max_epsilon
        self.batch_size = batch_size
        self.filtration = None

    def __batch_distances(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates the distances between points in data batches for memory efficiency.

        Parameters:
        -----------
        df : pl.DataFrame
            Dataframe of input features.

        Returns:
        --------
        pl.DataFrame
            A dataframe of edge distances.
        """

        distance_chunks = []
        n_samples = df.select(pl.len()).collect().item()
        for i in range(0, n_samples, self.batch_size):
            chunk_a = df.slice(i, self.batch_size)

            # Only compare against chunks further down to avoid redundant pairs
            for j in range(i, n_samples, self.batch_size):
                chunk_b = df.slice(j, self.batch_size)

                # Cross join
                # To optimize, filter first where a.id < b.id to avoid self-distance
                pairs_chunk = chunk_a.join(
                    chunk_b, how="cross", suffix="_right"
                ).filter(pl.col("id") < pl.col("id_right"))

                # Calculate Euclidean distance and filter by max_epsilon
                distance_expr = pl.sum_horizontal(
                    [
                        (pl.col(c) - pl.col(f"{c}_right")).pow(2)
                        for c in df.collect_schema().names()
                        if c not in ["id", "id_right"]
                    ]
                ).sqrt()

                distance_chunk = (
                    pairs_chunk.with_columns(distance_expr.alias("distance"))
                    .filter(pl.col("distance") <= self.max_epsilon)
                    .select(["id", "id_right", "distance"])
                )

                # Check if distance_chunk is empty before appending
                is_empty = distance_chunk.select(pl.len()).collect().item() == 0
                if not is_empty:
                    distance_chunks.append(distance_chunk)

        # Stitch just the valid sparse edges together
        distances = pl.concat(distance_chunks).collect(streaming=True)
        return distances

    def fit_transform(self, df: pl.DataFrame) -> None:
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

        # Retrieve distance matrix
        edges_base = self.__batch_distances(df=df_with_id)

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
        self.all_simplices_df = (
            pl.concat(all_dfs, how="diagonal")
            .lazy()
            .sort(["birth_time", "dim"])
            .collect(streaming=True)
        )

    def get_all_simplices(self) -> list:
        """
        Retrieves all simplices in the filtration.

        Returns:
        --------
        list of tuples
            A sorted list of simplices forming the filtration.
            Each element is a tuple: (simplex_tuple, birth_time)
            where simplex_tuple is a tuple of vertex indices (sorted) and birth_time is a float.
        """

        if self.all_simplices_df is None:
            raise ValueError(
                "Filtration has not been computed. Call fit_transform() first."
            )

        self.filtration = []
        vertex_cols = [c for c in self.all_simplices_df.columns if c.startswith("v_")]
        for row in self.all_simplices_df.select(
            [*vertex_cols, "birth_time"]
        ).to_dicts():
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

        if self.all_simplices_df is None:
            raise ValueError(
                "Filtration has not been computed. Call fit_transform() first."
            )

        return [
            (tuple(row[f"v_{i}"] for i in range(dim + 1)), row["birth_time"])
            for row in self.all_simplices_df.filter(pl.col("dim") == dim).to_dicts()
        ]

    def compute_birth_death_pairs(self):
        """
        Computes birth-death pairs from the filtration.

        Returns:
        --------
        persistence_pairs : dict
            A dictionary mapping dimension -> list of (birth_time, death_time) tuples.
        """

        if self.all_simplices_df is None:
            raise ValueError(
                "Filtration has not been computed. Call fit_transform() first."
            )

        num_simplices = self.all_simplices_df.height
        self.persistence_pairs = {}

        # Pre-extract all simplex metadata
        birth_times = (
            self.all_simplices_df.select(pl.col("birth_time")).to_numpy().flatten()
        )
        dimensions = self.all_simplices_df.select(pl.col("dim")).to_numpy().flatten()

        # Store vertices as a 2D matrix where missing entries are padded with -1
        v_cols = [c for c in self.all_simplices_df.columns if c.startswith("v_")]
        vertices_matrix = self.all_simplices_df.select(
            [pl.col(c).fill_null(-1).cast(pl.Int64) for c in v_cols]
        ).to_numpy()

        # Build a fast reverse lookup for simplex mapping
        simplex_to_idx = {
            tuple(v for v in row if v != -1): i for i, row in enumerate(vertices_matrix)
        }

        # Calculate the cohomology boundary matrix
        coboundaries = [[] for _ in range(num_simplices)]
        for col_idx in range(num_simplices):
            dim = dimensions[col_idx]
            # Vertices of the current simplex
            simplex = [v for v in vertices_matrix[col_idx] if v != -1]

            if dim > 0:
                # Homology looks at its faces. Cohomology uses faces to tell the lower-dimensional face that this simplex is part of its coboundary.
                for i in range(len(simplex)):
                    face = tuple(simplex[:i] + simplex[i + 1 :])
                    face_idx = simplex_to_idx.get(face, -1)
                    if face_idx != -1:
                        # Append the current simplex as a cofacet of its face
                        coboundaries[face_idx].append(col_idx)

        for i in range(num_simplices):
            coboundaries[i].sort()

        # Dictionary that maps a pivot row to its column vector
        cocycle_basis = {}
        pivot_to_row = np.full(num_simplices, -1, dtype=np.int32)
        # Track columns that are mathematically guaranteed to reduce to zero
        cleared = np.zeros(num_simplices, dtype=np.bool_)

        # Core coboundary matrix reduction Loop
        for col_idx in range(num_simplices):
            dim = dimensions[col_idx]

            # Skip max dimension because it cannot destroy anything or have a coboundary
            if dim == self.max_dim:
                continue

            # If this column was marked as cleared by a higher dimension, skip its reduction entirely!
            # Its boundary becomes implicitly empty.
            if cleared[col_idx]:
                continue

            # Initialize a new active cocycle tracking vector containing just this simplex
            current_cocycle = {col_idx}
            # Find the lowest un-eliminated entry in the coboundary matrix
            # entries are checked dynamically
            coboundary_elements = list(coboundaries[col_idx])

            while coboundary_elements:
                # Pivot for cohomology is the FIRST (earliest entering) element in the coboundary
                pivot_row = coboundary_elements[0]

                # Check if this pivot has already been claimed by a different cocycle
                other_col = pivot_to_row[pivot_row]

                if other_col != -1:
                    # XOR/Symmetric difference with the existing claimed cocycle's coboundary
                    # This eliminates the leading pivot element
                    current_cocycle = current_cocycle ^ cocycle_basis[other_col]

                    # Recompute the active coboundary elements of the combined cocycle
                    new_coboundary = set()
                    for c_idx in current_cocycle:
                        new_coboundary = new_coboundary ^ set(coboundaries[c_idx])

                    coboundary_elements = sorted(list(new_coboundary))
                else:
                    # A valid pair is found
                    pivot_to_row[pivot_row] = col_idx
                    cocycle_basis[col_idx] = current_cocycle

                    # Clearing property: since pivot_row is a creator paired with a destroyer,
                    # its own column is guaranteed to reduce to zero. Mark it to be skipped!
                    cleared[pivot_row] = True

                    birth_time = birth_times[col_idx]
                    death_time = birth_times[pivot_row]
                    birth_dim = dim

                    if birth_dim not in self.persistence_pairs:
                        self.persistence_pairs[birth_dim] = []

                    if death_time > birth_time:
                        self.persistence_pairs[birth_dim].append(
                            (birth_time, death_time)
                        )
                    break

        # Collect essential features (unpaired cycles that live to infinity)
        for col_idx in range(num_simplices):
            if (
                (dimensions[col_idx] < self.max_dim)
                and (not cleared[col_idx])
                and (col_idx not in cocycle_basis)
            ):
                dim = dimensions[col_idx]
                if dim not in self.persistence_pairs:
                    self.persistence_pairs[dim] = []
                self.persistence_pairs[dim].append((birth_times[col_idx], float("inf")))

        return self.persistence_pairs
