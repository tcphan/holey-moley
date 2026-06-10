import numpy as np

class PersistentHomology:
    """
    Computes the persistent homology of a simplicial complex filtration using boundary matrix reduction.
    """
    def __init__(self):
        pass

    def fit_transform(self, filtration):
        """
        Computes the persistence intervals from a sorted filtration.

        Parameters:
        -----------
        filtration : list of tuples
            A sorted list of simplices, where each element is: (simplex_tuple, birth_time)

        Returns:
        --------
        intervals : dict
            A dictionary where keys are dimensions (0, 1, ...) and values are lists of 
            tuples (birth, death) representing persistence intervals.
        """
        # Map each simplex tuple to its index in the filtration
        simplex_to_idx = {simplex: idx for idx, (simplex, _) in enumerate(filtration)}
        M = len(filtration)
        
        # Build the boundary matrix as list of sets containing non-zero row indices (Z2 coefficients)
        D = [set() for _ in range(M)]
        for j in range(M):
            simplex, _ = filtration[j]
            dim = len(simplex) - 1
            if dim > 0:
                # Boundary of a p-simplex consists of its (p-1)-dimensional faces
                for i in range(len(simplex)):
                    face = simplex[:i] + simplex[i+1:]
                    # Face must exist in the filtration
                    if face in simplex_to_idx:
                        D[j].add(simplex_to_idx[face])

        # pivot_to_col maps a row index (pivot) to the column index that has it as lowest non-zero element
        pivot_to_col = {}
        
        # Reduced boundary matrix column storage
        R = [D[j].copy() for j in range(M)]

        # Perform boundary matrix reduction (standard algorithm over Z2)
        for j in range(M):
            while R[j]:
                pivot = max(R[j])  # lowest non-zero row in column j (since indices are ordered)
                if pivot in pivot_to_col:
                    # Add (XOR) the column that already has this pivot to column j
                    k = pivot_to_col[pivot]
                    R[j] ^= R[k]  # Symmetric difference is equivalent to addition mod 2
                else:
                    pivot_to_col[pivot] = j
                    break

        # Group intervals by dimension
        # intervals: dim -> list of (birth_time, death_time)
        intervals = {}

        # Set of columns that are pivots (deaths)
        death_columns = set(pivot_to_col.values())

        for j in range(M):
            simplex, birth_time = filtration[j]
            dim = len(simplex) - 1
            
            if dim not in intervals:
                intervals[dim] = []

            # If column j reduced to a pivot, it means it is a death simplex
            # and it paired with a birth simplex (which was the pivot)
            # We already handle pairs from the perspective of birth/death
            
            # If j is in pivot_to_col:
            # j is a death column that kills the feature born at row `pivot`
            # The simplex at row `pivot` represents the birth simplex.
            # Thus, we pair birth simplex (filtration[pivot]) with death simplex (filtration[j]).
            # Note: the homology class has the dimension of the birth simplex.
            
        # We can reconstruct all pairs:
        for pivot, j in pivot_to_col.items():
            birth_simplex, birth_time = filtration[pivot]
            death_simplex, death_time = filtration[j]
            birth_dim = len(birth_simplex) - 1
            
            if birth_dim not in intervals:
                intervals[birth_dim] = []
            
            # Only record if death_time is greater than or equal to birth_time
            # (though theoretically filtration ensures death_time >= birth_time)
            intervals[birth_dim].append((birth_time, death_time))

        # Add infinite persistence intervals:
        # A column j that reduced to empty (R[j] is empty) and is NOT a pivot (not in pivot_to_col)
        # represents a feature born at simplex j that never dies.
        for j in range(M):
            if not R[j] and j not in pivot_to_col:
                simplex, birth_time = filtration[j]
                dim = len(simplex) - 1
                
                if dim not in intervals:
                    intervals[dim] = []
                
                intervals[dim].append((birth_time, float('inf')))

        # Sort intervals by birth time for clean output
        for dim in intervals:
            intervals[dim] = sorted(intervals[dim], key=lambda x: (x[0], x[1]))

        return intervals
