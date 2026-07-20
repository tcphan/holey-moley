import numpy as np
from scipy.optimize import linear_sum_assignment
import polars as pl
import polars.selectors as cs


def check_bipartite_matching(threshold, distance_df, num_a, num_b):
    """
    Verifies if a valid bottleneck matching exists for a given threshold using Maximum Bipartite Matching.

    Parameters:
    -----------
    threshold : float
        The threshold value to check for.
    distance_df : pl.DataFrame
        The distance dataframe.
    num_a : int
        The number of points in the first diagram.
    num_b : int
        The number of points in the second diagram.

    Returns:
    --------
    bool
        True if a valid bottleneck matching exists, False otherwise.
    """

    # Filter for edges that are valid under this threshold
    # A pair is valid if their direct distance <= threshold OR if both can go to the diagonal
    valid_edges = distance_df.filter(
        (pl.col("l_inf_distance") <= threshold)
        | (
            pl.max_horizontal(["distance_to_diagonal_a", "distance_to_diagonal_b"])
            <= threshold
        )
    )

    # Extract distances to diagonal for unmatched points verification
    # We group by to get unique IDs and their diagonal distances
    diag_a_lookup = dict(
        distance_df.select(["id_a", "distance_to_diagonal_a"]).unique().iter_rows()
    )
    diag_b_lookup = dict(
        distance_df.select(["id_b", "distance_to_diagonal_b"]).unique().iter_rows()
    )
    num_a = len(diag_a_lookup)
    num_b = len(diag_b_lookup)

    # Build an adjacency list from the filtered Polars data
    adj = {i: [] for i in range(num_a)}
    for row in valid_edges.select(["id_a", "id_b"]).iter_rows():
        adj[row[0]].append(row[1])

    # Bipartite matching algorithm (Hungarian algorithm style)
    match_b = [-1] * num_b

    def can_match(u, visited):
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_b[v] < 0 or can_match(match_b[v], visited):
                    match_b[v] = u
                    return True
        return False

    # Compute maximum matching
    for u in range(num_a):
        visited = [False] * num_b
        can_match(u, visited)

    # Validate unmatched points: Any point left unmatched MUST be close enough to the diagonal
    match_a = [-1] * num_a
    for v, u in enumerate(match_b):
        if u != -1:
            match_a[u] = v

    # Check Diagram A unmatched points
    for u in range(num_a):
        if match_a[u] == -1 and diag_a_lookup[u] > threshold:
            return False

    # Check Diagram B unmatched points
    for v in range(num_b):
        if match_b[v] == -1 and diag_b_lookup[v] > threshold:
            return False

    return True


def bottleneck_distance(
    diagram_a: list[tuple[float, float]], diagram_b: list[tuple[float, float]]
) -> float:
    """
    Calculates the bottleneck distance between two persistence diagrams.

    Parameters:
    -----------
    diagram_a : list of tuples
        The first persistence diagram. Each tuple is a pair of (birth_time, death_time).
    diagram_b : list of tuples
        The second persistence diagram. Each tuple is a pair of (birth_time, death_time).

    Returns:
    --------
    float
        The bottleneck distance between the two persistence diagrams.
    """

    num_a = len(diagram_a)
    num_b = len(diagram_b)

    # Handle empty diagrams
    if num_a == 0 and num_b == 0:
        return 0.0

    if num_a == 0 or num_b == 0:
        return float("inf")

    # Convert diagram to dataframe
    schema = ["birth_time", "death_time"]
    diagram_a_df = pl.DataFrame(diagram_a, schema=schema, orient="row").with_row_index(
        "id"
    )
    diagram_b_df = pl.DataFrame(diagram_b, schema=schema, orient="row").with_row_index(
        "id"
    )

    # Calculate the L-infinity distance between each point to the diagonal
    dist_p_to_diagonal = (pl.col("death_time") - pl.col("birth_time")) / 2
    diagram_a_df = diagram_a_df.with_columns(
        dist_p_to_diagonal.alias("distance_to_diagonal")
    )
    diagram_b_df = diagram_b_df.with_columns(
        dist_p_to_diagonal.alias("distance_to_diagonal")
    )

    # Cross join: filter first where a.id <= b.id to avoid duplicate pairs
    diagram_a_df = diagram_a_df.select(cs.all().name.suffix("_a"))
    diagram_b_df = diagram_b_df.select(cs.all().name.suffix("_b"))
    joined_df = diagram_a_df.join(diagram_b_df, how="cross")

    # Calculate the L-infinity distance between all pairs of points
    joined_df = joined_df.with_columns(
        pl.max_horizontal(
            [
                abs(pl.col("birth_time_a") - pl.col("birth_time_b")),
                abs(pl.col("death_time_a") - pl.col("death_time_b")),
            ]
        ).alias("l_inf_distance")
    )

    # Gather unique candidate thresholds from our calculations
    # The bottleneck distance is always explicitly one of these exact values
    candidates_pair = joined_df.select("l_inf_distance").to_series().to_list()
    candidates_diag_a = joined_df.select("distance_to_diagonal_a").to_series().to_list()
    candidates_diag_b = joined_df.select("distance_to_diagonal_b").to_series().to_list()

    thresholds_list = sorted(
        list(set(candidates_pair + candidates_diag_a + candidates_diag_b))
    )

    # Binary search over thresholds using our bipartite matching checker
    low = 0
    high = len(thresholds_list) - 1
    ans = thresholds_list[high] if thresholds_list else 0.0

    while low <= high:
        mid = (low + high) // 2
        current_threshold = thresholds_list[mid]

        if check_bipartite_matching(current_threshold, joined_df, num_a, num_b):
            ans = current_threshold
            high = mid - 1  # Try to find a smaller working bottleneck distance
        else:
            low = mid + 1  # Threshold too small, increase it

    return ans


def wasserstein_distance(
    diagram_a: list[tuple[float, float]],
    diagram_b: list[tuple[float, float]],
    p: float = 1.0,
    l_distance_metric: str = "l_inf",
) -> float:
    """
    Calculates the Wasserstein (Earth's mover) distance between two persistence diagrams.

    Parameters:
    -----------
    diagram_a : list of tuples
        The first persistence diagram. Each tuple is a pair of (birth_time, death_time).
    diagram_b : list of tuples
        The second persistence diagram. Each tuple is a pair of (birth_time, death_time).
    p : float
        The power exponent for the Wasserstein metric (p >= 1).
    l_distance_metric : str
        The distance metric to use for matching pairs. Options include: 'l_inf' (L-infinity) or 'l_2' (Euclidean).
        Defaults to "l_inf".

    Returns:
    --------
    float
        The Wasserstein distance between the two persistence diagrams.
    """

    num_a = len(diagram_a)
    num_b = len(diagram_b)

    # Handle empty diagrams
    if num_a == 0 and num_b == 0:
        return 0.0

    if num_a == 0 or num_b == 0:
        return float("inf")

    # Convert to DataFrames and compute diagonal projection distances
    schema = ["birth", "death"]
    df_a = pl.DataFrame(diagram_a, schema=schema, orient="row").with_row_index("id")
    df_b = pl.DataFrame(diagram_b, schema=schema, orient="row").with_row_index("id")

    if l_distance_metric == "l_inf":
        diag_expr = (pl.col("death") - pl.col("birth")) / 2.0
    elif l_distance_metric == "l_2":
        diag_expr = (pl.col("death") - pl.col("birth")) / pl.lit(2.0).sqrt()
    else:
        raise ValueError("l_distance_metric must be either 'l_inf' or 'l_2'")

    df_a = df_a.with_columns(diag_expr.alias("dist_to_diag"))
    df_b = df_b.with_columns(diag_expr.alias("dist_to_diag"))

    size = num_a + num_b
    cost_matrix = np.full((size, size), np.inf)

    # Block 1 (Top-Left): Pairwise costs between Diagram A and Diagram B
    if num_a > 0 and num_b > 0:
        # Cross join to get full mapping grid
        grid = df_a.select(cs.all().name.suffix("_a")).join(
            df_b.select(cs.all().name.suffix("_b")), how="cross"
        )

        if l_distance_metric == "l_inf":
            dist_expr = pl.max_horizontal(
                [
                    (pl.col("birth_a") - pl.col("birth_b")).abs(),
                    (pl.col("death_a") - pl.col("death_b")).abs(),
                ]
            )
        else:
            dist_expr = (
                (pl.col("birth_a") - pl.col("birth_b")).pow(2)
                + (pl.col("death_a") - pl.col("death_b")).pow(2)
            ).sqrt()

        # Calculate cost ^ p and pivot into an adjacency matrix format
        pairwise_costs = (
            grid.with_columns((dist_expr**p).alias("cost"))
            .pivot(on="id_b", index="id_a", values="cost")
            .drop("id_a")
        )

        cost_matrix[:num_a, :num_b] = pairwise_costs.to_numpy()

    # Block 2 (Top-Right): Diagram A matched to their corresponding diagonal slots
    if num_a > 0:
        costs_diag_a = (df_a.select("dist_to_diag").to_series() ** p).to_numpy()
        # Each point A_i maps uniquely to a dummy diagonal partner at index (num_b + i)
        cost_matrix[np.arange(num_a), num_b + np.arange(num_a)] = costs_diag_a

    # Block 3 (Bottom-Left): Diagram B matched to their corresponding diagonal slots
    if num_b > 0:
        costs_diag_b = (df_b.select("dist_to_diag").to_series() ** p).to_numpy()
        # Each point B_j maps uniquely to a dummy diagonal partner at index (num_a + j)
        cost_matrix[num_a + np.arange(num_b), np.arange(num_b)] = costs_diag_b

    # Block 4 (Bottom-Right): Diagonal-to-Diagonal interactions
    # Remain initialized as 0.0 so that surplus unmatched elements add nothing to overall cost
    cost_matrix[num_a:, num_b:] = 0.0

    # Solve matching via linear sum assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return float(total_cost ** (1.0 / p))
