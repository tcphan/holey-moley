import numpy as np
from scipy.spatial.distance import cdist

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

    def fit_transform(self, X, is_distance_matrix=False):
        """
        Computes the Vietoris-Rips filtration for the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Input point cloud or distance matrix.
        is_distance_matrix : bool, default=False
            If True, X is assumed to be a pairwise distance matrix. Otherwise, 
            it is assumed to be a point cloud.

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
            dist_matrix = cdist(X, X, metric='euclidean')

        n_samples = dist_matrix.shape[0]
        
        # Dictionary to store simplices by dimension
        # Key: dimension, Value: list of tuples (simplex, birth_time)
        simplices_by_dim = {}
        
        # Dimension 0: Vertices
        simplices_by_dim[0] = [((i,), 0.0) for i in range(n_samples)]
        
        # Iteratively build higher dimensions
        for d in range(1, self.max_dim + 1):
            simplices_by_dim[d] = []
            # Generate d-simplices from (d-1)-simplices
            for face, face_birth in simplices_by_dim[d-1]:
                last_v = face[-1]
                for u in range(last_v + 1, n_samples):
                    # Check if all edges between u and face vertices are within max_epsilon
                    valid = True
                    max_edge = face_birth
                    for v in face:
                        dist = dist_matrix[u, v]
                        if dist > self.max_epsilon:
                            valid = False
                            break
                        if dist > max_edge:
                            max_edge = dist
                    
                    if valid:
                        new_simplex = face + (u,)
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

        filtration = sorted(all_simplices, key=filtration_key)
        return filtration
