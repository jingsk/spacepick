import numpy as np

class Selector:
    """
    A tool for subset selection using a relative vector matrix (M x M x 3).
    Supports dispersed, clustered, and directional linear selection.
    """
    def __init__(self, rel_vectors):
        """
        Args:
            rel_vectors (np.ndarray): Shape (M, M, 3) where entry [i, j] 
                                      is the vector pointing from i to j.
        """
        self.rel_vectors = np.asarray(rel_vectors)
        self.m = self.rel_vectors.shape[0]
        
        # Precompute scalar distance matrix for dispersed/clustered modes
        self.dist_matrix = np.linalg.norm(self.rel_vectors, axis=2)
        
        # Map axis names to indices
        self.axis_map = {'x': 0, 'y': 1, 'z': 2}

    def get_indices(self, n_subset, mode='dispersed', refine=True, direction_pref='yx', tol=1e-3):
        """
        Args:
            n_subset (int): Number of points to select.
            mode (str): 'dispersed', 'clustered', or 'linear'.
            refine (bool): Run swap heuristic (only for dispersed/clustered).
            direction_pref (str): Priority of axes for 'linear' mode (e.g., 'yx').
        """
        if mode == 'dispersed':
            indices = self._greedy_dispersed(n_subset)
        elif mode == 'clustered':
            indices = self._greedy_clustered(n_subset)
        elif mode == 'linear':
            # Refinement isn't typically used for linear as it follows a path
            return self._greedy_linear(n_subset, direction_pref, tol)
        else:
            raise ValueError("Mode must be 'dispersed', 'clustered', or 'linear'.")
            
        if refine:
            indices = self._refine_swap(indices, mode)
        return indices

    def _greedy_linear(self, n, pref, tol):
        """
        Strict directional traversal. 
        Prioritizes points directly on the axis of the primary preference.
        """
        #tol = np.asarray(tol)
        if len(tol) == 1:
            tol = 2 * tol
        pref_axes = [self.axis_map[c] for c in pref.lower()]
        all_axes = [0, 1, 2]
        
        selected = [0]
        remaining = set(range(1, self.m))

        for _ in range(n - 1):
            curr = selected[-1]
            best_neighbor = None
            
            # Iterate through preference (e.g., first 'y', then 'x')
            for target_axis, t in zip(pref_axes, tol):
                # Identify 'other' axes that should be zero for strict collinearity
                other_axes = [a for a in all_axes if a != target_axis]
                
                candidates = []
                for idx in remaining:
                    vec = self.rel_vectors[curr, idx]
                    
                    # Condition 1: Must be in the positive direction of target axis
                    # Condition 2: Must be "directly along" (other components ~ 0)
                    is_ahead = vec[target_axis] > t
                    is_collinear = all(abs(vec[oa]) < t for oa in other_axes)
                    
                    if is_ahead and is_collinear:
                        candidates.append(idx)
                
                if candidates:
                    # Pick the closest one among those directly along the axis
                    # to ensure we don't skip steps in the grid
                    best_neighbor = candidates[np.argmin(self.dist_matrix[curr, candidates])]
                    break
            
            # If no strict collinear neighbor is found in any preferred direction
            if best_neighbor is None and remaining:
                # Fallback: pick the closest point regardless of direction 
                # to keep the subset connected
                candidates = list(remaining)
                best_neighbor = candidates[np.argmin(self.dist_matrix[curr, candidates])]

            if best_neighbor is not None:
                selected.append(best_neighbor)
                remaining.remove(best_neighbor)
            else:
                break
        return selected
        
    def _get_score(self, indices, mode):
        """Calculates the metric to optimize based on the mode."""
        if len(indices) < 2: return 0
        sub_matrix = self.dist_matrix[np.ix_(indices, indices)]
        
        if mode == 'dispersed':
            # Score = Minimum pairwise distance (Goal: Maximize this)
            mask = np.eye(len(indices), dtype=bool)
            return np.min(sub_matrix[~mask])
        else:
            # Score = Maximum pairwise distance (Goal: Minimize this)
            return np.max(sub_matrix)
    def _greedy_dispersed(self, n):
        """Farthest-First Traversal."""
        selected = [0]
        min_dists = np.copy(self.dist_matrix[0, :])
        for _ in range(1, n):
            next_idx = np.argmax(min_dists)
            selected.append(next_idx)
            min_dists = np.minimum(min_dists, self.dist_matrix[next_idx, :])
        return selected

    def _greedy_clustered(self, n):
        """Picks the point with the smallest 'radius' to its neighbors."""
        # Start with the point that has the smallest average distance to everyone else
        start_idx = np.argmin(np.mean(self.dist_matrix, axis=1))
        selected = [start_idx]
        
        for _ in range(1, n):
            # Find point that minimizes the maximum distance to any already selected point
            candidates = list(set(range(self.m)) - set(selected))
            # Max distance from each candidate to the selected set
            max_dists_to_set = np.max(self.dist_matrix[selected][:, candidates], axis=0)
            next_idx = candidates[np.argmin(max_dists_to_set)]
            selected.append(next_idx)
        return selected

    def _refine_swap(self, indices, mode):
        """Iteratively swap points to improve the subset score."""
        current_indices = list(indices)
        current_score = self._get_score(current_indices, mode)
        all_pool = set(range(self.m))
        
        improved = True
        while improved:
            improved = False
            unselected = list(all_pool - set(current_indices))
            
            for i in range(len(current_indices)):
                for j in unselected:
                    candidate = current_indices.copy()
                    candidate[i] = j
                    score = self._get_score(candidate, mode)
                    
                    if mode == 'dispersed' and score > current_score:
                        current_score, current_indices, improved = score, candidate, True
                    elif mode == 'clustered' and score < current_score:
                        current_score, current_indices, improved = score, candidate, True
                    
                    if improved: break
                if improved: break
        return current_indices

