import numpy as np
import polars as pl


def persistence_landscape(
    persistence_pairs: dict[int, list[tuple[float, float]]],
    dim: int = 1,
    num_landscapes: int = 3,
    num_eval_points: int = 100,
    t_min: float = None,
    t_max: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the persistence landscapes for a specified dimension.

    Parameters:
    -----------
    dim : int
        The homology dimension to build landscapes for (e.g., 0 for clusters, 1 for loops).
    num_landscapes : int
        The number of top landscape layers (k=1, 2, ..., K) to return.
    num_eval_points : int
        Resolution (number of evaluation points) along the filtration axis t.
    t_min : float, optional
        The minimum bound of the evaluation domain. Defaults to the min birth time.
    t_max : float, optional
        The maximum bound of the evaluation domain. Defaults to the max finite death time.

    Returns:
    --------
    t_grid : np.ndarray
        1D array of shape (num_eval_points) containing the evaluation points.
    landscapes : np.ndarray
        2D array of shape (num_landscapes, num_eval_points) containing the k-th landscape functions.
    """

    # Filter out infinite features or pairs where death <= birth
    pairs = persistence_pairs.get(dim, [])
    valid_pairs = [
        (b, d) for b, d in pairs if d != float("inf") and not np.isinf(d) and d > b
    ]

    # If no features exist in this dimension, return zero landscapes
    if not valid_pairs:
        t_min = 0.0 if t_min is None else t_min
        t_max = 1.0 if t_max is None else t_max
        t_grid = np.linspace(t_min, t_max, num_eval_points)
        return t_grid, np.zeros((num_landscapes, num_eval_points))

    # Auto-infer grid boundaries if not explicitly provided
    if t_min is None:
        t_min = min(b for b, d in valid_pairs)
    if t_max is None:
        t_max = max(d for b, d in valid_pairs)

    # Create evaluation grid
    t_vals = pl.linear_space(t_min, t_max, num_eval_points, eager=True).to_list()
    t_grid = pl.DataFrame(
        {
            "birth": [b for b, d in valid_pairs],
            "death": [d for b, d in valid_pairs],
            **{f"t_ind_{i}": [val] * len(valid_pairs) for i, val in enumerate(t_vals)},
        }
    )

    # Compute tent function f_{(b,d)}(t) for each birth-death pair over the grid
    # f_{(b,d)}(t) = max(0, min(t - b, d - t))
    t_grid = t_grid.with_columns(
        pl.max_horizontal(
            [
                pl.lit(0.0),
                pl.min_horizontal(
                    [
                        pl.col(f"t_ind_{i}") - pl.col("birth"),
                        pl.col("death") - pl.col(f"t_ind_{i}"),
                    ]
                ),
            ]
        ).alias(f"tent_{i}")
        for i in range(num_eval_points)
    )

    # Sort each tent column in descending order to obtain Order Statistics
    t_grid = t_grid.with_columns(
        [
            pl.col(f"tent_{i}").sort(descending=True).alias(f"tent_{i}")
            for i in range(num_eval_points)
        ]
    )

    # Fill available layers up to requested num_landscapes
    num_available = len(valid_pairs)
    k_to_fill = min(num_landscapes, num_available)

    # Extract top k_to_fill layers
    top_layers = t_grid.head(k_to_fill).select(
        [f"tent_{i}" for i in range(num_eval_points)]
    )
    landscapes = top_layers.to_numpy()

    # Pad with zero layers if num_landscapes > available valid pairs
    if num_landscapes > num_available:
        padding = np.zeros((num_landscapes - num_available, num_eval_points))
        landscapes = np.vstack([landscapes, padding])

    return (np.array(t_vals), landscapes)
