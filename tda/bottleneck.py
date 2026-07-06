import numpy as np
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
