import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

@njit
def _compute_edges(dist_matrix, max_epsilon):
    """
    Calculates the edges of the Rips complex.

    Parameters:
    -----------
    dist_matrix : np.ndarray
        The distance matrix of the point cloud.
    max_epsilon : float
        The maximum filtration value (threshold distance) for adding simplices.
    
    Returns:
    --------
    from_nodes : list
        List of source nodes for each edge.
    to_nodes : list
        List of target nodes for each edge.
    distances : list
        List of distances for each edge.
    """
    n_samples = dist_matrix.shape[0]
    from_nodes = []
    to_nodes = []
    distances = []
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = dist_matrix[i, j]
            if d <= max_epsilon:
                from_nodes.append(i)
                to_nodes.append(j)
                distances.append(d)
                
    return from_nodes, to_nodes, distances

class VietorisRips:
    """
    Constructs the Vietoris-Rips filtration of a point cloud or distance matrix.
    """
    def __init__(self, max_dim=2, max_epsilon=float('inf')):
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

    def fit_transform(self, X, is_distance_matrix=False, distance_metric="euclidean"):
        """
        Computes the Vietoris-Rips filtration for the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Input point cloud or distance matrix.
        is_distance_matrix : bool, default=False
            If True, X is assumed to be a pairwise distance matrix. Otherwise, 
            it is assumed to be a point cloud.
        distance_metric : str, default="euclidean"
            The metric to use for distance calculations.

        Returns:
        --------
        filtration : list of tuples
            A sorted list of simplices forming the filtration.
            Each element is a tuple: (simplex_tuple, birth_time)
            where simplex_tuple is a tuple of vertex indices (sorted) and birth_time is a float.
        """
        X = np.asarray(X, dtype=float)
        if is_distance_matrix:
            dist_matrix = X
        else:
            dist_matrix = cdist(X, X, metric=distance_metric)

        n_samples = dist_matrix.shape[0]

        # Use Numba-accelerated helper function for the heavy O(N^2) distance check loop
        from_nodes, to_nodes, distances = _compute_edges(dist_matrix, self.max_epsilon)

        # Pre-calculate an adjacency list of neighbors for each vertex
        # Only keep neighbors where u < v to avoid double-counting, and within max_epsilon
        adj_list = {i: set() for i in range(n_samples)}
        edges = []

        for u, v, d in zip(from_nodes, to_nodes, distances):
            adj_list[u].add(v)
            edges.append(((u, v), d))
        
        # Dictionary to store simplices by dimension
        # Key: dimension, Value: list of tuples (simplex, birth_time)
        simplices_by_dim = {}
        
        # Dimension 0: Vertices
        simplices_by_dim[0] = [((i,), 0.0) for i in range(n_samples)]
        # Dimension 1: Edges
        simplices_by_dim[1] = edges

        # Inductive expansion using intersection of neighbor sets
        for d in range(2, self.max_dim + 1):
            simplices_by_dim[d] = []
            # Generate d-simplices from (d-1)-simplices
            for face, face_birth in simplices_by_dim[d-1]:
                # Find the common neighbors of ALL vertices in the current face
                # We start with the neighbors of the first vertex
                common_neighbors = adj_list[face[0]]
                for v in face[1:]:
                    common_neighbors = common_neighbors.intersection(adj_list[v])
                
                # Filter to ensure we only look at vertices greater than the last vertex
                # to maintain lexicographical order and avoid duplicates
                last_v = face[-1]
                valid_extensions = [u for u in common_neighbors if u > last_v]
                
                for u in valid_extensions:
                    new_simplex = face + (u,)
                    # The birth time is the maximum distance between the new vertex u and existing face vertices
                    max_edge = max(face_birth, max(dist_matrix[u, v] for v in face))
                    simplices_by_dim[d].append((new_simplex, max_edge))

        # Flatten all simplices into a single list
        all_simplices = []
        for d in range(self.max_dim + 1):
            all_simplices.extend(simplices_by_dim[d])

        # Sort all simplices to form a valid filtration:
        # 1. By birth time
        # 2. By dimension (number of vertices) to break ties (faces before cofaces)
        # 3. Lexicographically by vertex indices
        def filtration_key(item):
            simplex, birth_time = item
            return (birth_time, len(simplex), simplex)

        self.filtration = sorted(all_simplices, key=filtration_key)
        return self.filtration

    def _create_boundary_matrix(self, simplex_to_idx):
        """
        Compute the boundary matrix for the filtration.
        
        Parameters:
        -----------
        simplex_to_idx : dict
            A dictionary mapping each simplex to its index in the filtration.
        
        Returns:
        --------
        boundary_matrix : np.ndarray
            The boundary matrix where entry (i, j) is 1 if the j-th simplex is a 
            face of the i-th simplex, and 0 otherwise.
        """

        simplices = [simplex for simplex, _ in simplex_to_idx.items()]

        # Identify the boundaries that make up each simplex
        boundaries = []
        for col_idx in range(len(simplices)):
            simplex = simplices[col_idx]
            dim = len(simplex) - 1
            if dim == 0:
                boundaries.append(np.array([], dtype=np.int32))
            else:
                faces = []
                for i in range(len(simplex)):
                    face = simplex[:i] + simplex[i+1:]
                    # Safe lookup
                    idx = simplex_to_idx.get(face, -1)
                    if idx != -1:
                        faces.append(idx)
                # Sort descending so the first element is always the current pivot candidate
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
        intervals = {}

        # pivot_to_col[r] stores the column index c that has its pivot at row r (-1 means unassigned)
        pivot_to_col = np.full(num_simplices, -1, dtype=np.int32)
        # is_cycle represents if a simplex is a cycle (has no pivot)
        is_cycle = np.ones(num_simplices, dtype=np.bool_)

        # Pre-extract all simplex metadata into fast numpy arrays for O(1) loop checks
        birth_times = np.array([item[1] for item in self.filtration], dtype=np.float64)
        dimensions = np.array([len(item[0]) - 1 for item in self.filtration], dtype=np.int32)
        simplices = [item[0] for item in self.filtration]

        # Calculate the boundary matrix
        boundaries = self._create_boundary_matrix(simplex_to_idx)
        
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
                    boundary_indices = list(set(boundary_indices) ^ set(boundaries[other_col]))
                    boundary_indices.sort(reverse=True)
                else:
                    # pivot_row is born, col_idx kills it
                    pivot_to_col[pivot_row] = col_idx
                    # col_idx is now a boundary, not a cycle
                    is_cycle[col_idx] = False
                    
                    birth_dim = dimensions[pivot_row]
                    b_time = birth_times[pivot_row]
                    d_time = birth_times[col_idx] # The current simplex's birth time is the death time
                    
                    if birth_dim not in intervals:
                        intervals[birth_dim] = []
                    
                    if d_time > b_time:
                        intervals[birth_dim].append((b_time, d_time))
                    break

        # Collect essential features (unpaired cycles that live to infinity)
        for col_idx in range(num_simplices):
            if is_cycle[col_idx] and (pivot_to_col[col_idx] == -1):
                dim = dimensions[col_idx]
                if dim not in intervals:
                    intervals[dim] = []
                intervals[dim].append((birth_times[col_idx], float('inf')))

        return intervals
