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


def persistence_image(
    persistence_pairs: dict[int, list[tuple[float, float]]],
    dim: int = 1,
    bandwidth: float = 0.05,
    resolution: tuple[int, int] = (20, 20),
    b_range: tuple[float, float] = None,
    p_range: tuple[float, float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a persistence image from birth-death pairs.

    Parameters:
    -----------
    persistence_pairs : dict
        Dictionary mapping dimension -> list of (birth, death) tuples.
    dim : int
        Homology dimension to visualize.
    bandwidth : float
        Standard deviation (sigma) of the Gaussian kernel.
    resolution : tuple(int, int)
        Pixel grid resolution (nx, ny).
    b_range : tuple(float, float), optional
        Bounds for Birth axis (b_min, b_max). Auto-inferred if None.
    p_range : tuple(float, float), optional
        Bounds for Persistence axis (p_min, p_max). Auto-inferred if None.

    Returns:
    --------
    b_grid : np.ndarray
        1D array of birth grid bin centers.
    p_grid : np.ndarray
        1D array of persistence grid bin centers.
    image : np.ndarray
        2D persistence image matrix of shape (ny, nx).
    """

    nx, ny = resolution

    # Extract and transform (birth, death) -> (birth, persistence)
    pairs = persistence_pairs.get(dim, [])
    valid_data = [
        (b, d - b) for b, d in pairs if d != float("inf") and not np.isinf(d) and d > b
    ]

    if not valid_data:
        return (np.linspace(0, 1, nx), np.linspace(0, 1, ny), np.zero((ny, nx)))

    df_points = pl.DataFrame(valid_data, schema=["b_point", "p_point"])

    # Determine bounds for the image domain
    if b_range is None:
        b_min, b_max = (
            df_points["b_point"].min(),
            df_points["b_point"].max(),
        )
        # Add small buffer if min == max
        if b_min == b_max:
            b_min, b_max = b_min - 0.5, b_max + 0.5

    else:
        b_min, b_max = b_range

    if p_range is None:
        p_min, p_max = 0.0, df_points["p_point"].max()
        if p_min == p_max:
            p_max = 1.0
    else:
        p_min, p_max = p_range

    # Construct 1D grid centers
    b_centers = pl.linear_space(b_min, b_max, nx, eager=True).to_list()
    p_centers = pl.linear_space(p_min, p_max, ny, eager=True).to_list()

    # Create a 2D meshgrid dataframe of pixel centers (nx * ny rows)
    grid_df = pl.DataFrame(
        [
            (b, p)
            for p in p_centers  # Outer loop over y-axis (persistence)
            for b in b_centers  # Inner loop over x-axis (birth)
        ],
        schema=["b_pixel", "p_pixel"],
    )

    # Compute 2D Gaussian surface and weighting
    # Weighting functio w(b, p) = (p / p_max)^2 to downweight short-lived noise
    df_weighted_points = df_points.with_columns(
        ((pl.col("p_point") / p_max) ** 2).alias("weight")
    )

    # Cross join pixels with birth-persistence points to evaluate Gaussian densities
    # Density = sum_i [ weight_i * exp( -((b_px - b_i)^2 + (p_px - p_i)^2) / (2 * sigma^2) ) ]
    gaussian_expr = (
        pl.col("weight")
        * (
            -0.5
            * (
                ((pl.col("b_pixel") - pl.col("b_point")) / bandwidth) ** 2
                + ((pl.col("p_pixel") - pl.col("p_point")) / bandwidth) ** 2
            )
        ).exp()
    )

    image_df = (
        grid_df.join(df_weighted_points, how="cross")
        .with_columns(gaussian_expr.alias("density"))
        .group_by(["p_pixel", "b_pixel"], maintain_order=True)
        .agg(pl.col("density").sum())
    )

    # Reshape into 2D Matrix (ny, nx)
    image_matrix = image_df.select("density").to_numpy().reshape((ny, nx))

    return (np.array(b_centers), np.array(p_centers), image_matrix)
