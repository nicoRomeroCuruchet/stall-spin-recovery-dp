import numba as nb
import numpy as np


@nb.njit(cache=True)
def get_barycentric_weights_and_indices(
    points: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    grid_shape: np.ndarray,
    strides: np.ndarray,
    corner_bits: np.ndarray,
) -> tuple:
    """
    Calculate O(1) Barycentric interpolation weights and flat indices.

    Bypasses expensive algorithmic searches in favor of direct mathematical
    mapping for N-dimensional regular grids.
    """
    n_points, n_dims = points.shape
    n_corners = corner_bits.shape[0]

    weights = np.zeros((n_points, n_corners), dtype=np.float32)
    indices = np.zeros((n_points, n_corners), dtype=np.int32)

    step_sizes = (bounds_high - bounds_low) / (grid_shape - 1)

    for i in range(n_points):
        base_idx = np.zeros(n_dims, dtype=np.int32)
        t = np.zeros(n_dims, dtype=np.float32)

        for d in range(n_dims):
            # Clip points to bounds to avoid out-of-bounds indexing
            p = max(bounds_low[d], min(points[i, d], bounds_high[d]))

            # O(1) grid cell calculation
            cell = (p - bounds_low[d]) / step_sizes[d]
            idx_d = int(cell)

            # Handle edge case at the upper boundary
            if idx_d >= grid_shape[d] - 1:
                idx_d = grid_shape[d] - 2

            base_idx[d] = idx_d
            t[d] = (p - (bounds_low[d] + idx_d * step_sizes[d])) / step_sizes[d]

        # Compute weights and flat indices for all 2^N corners
        for c in range(n_corners):
            w = 1.0
            flat_idx = 0
            for d in range(n_dims):
                bit = corner_bits[c, d]
                w *= t[d] if bit else (1.0 - t[d])
                flat_idx += (base_idx[d] + bit) * strides[d]

            weights[i, c] = w
            indices[i, c] = flat_idx

    return weights, indices


def get_optimal_action(
    state: np.ndarray,
    optimal_policy: any,
    current_action_idx: int = None,
    hysteresis: float = 0.1,
) -> tuple:
    """
    Approximate the optimal action for a given state using barycentric interpolation.

    Computes a convex combination of the actions voted by the 16 surrounding vertices,
    weighted by their barycentric weights. This produces a continuous action that
    eliminates discontinuities at grid cell boundaries.

    Returns:
        (optimal_action, probabilities, None)
    """
    state_2d = np.atleast_2d(state).astype(np.float32)

    lambdas, points_indexes = get_barycentric_weights_and_indices(
        state_2d,
        optimal_policy.bounds_low,
        optimal_policy.bounds_high,
        optimal_policy.grid_shape,
        optimal_policy.strides,
        optimal_policy.corner_bits,
    )

    lambdas = lambdas.flatten()
    flat_indices = points_indexes.flatten()

    best_action_indices = optimal_policy.policy[flat_indices]

    n_neighbors = len(flat_indices)
    relevant_policies = np.zeros(
        (n_neighbors, optimal_policy.n_actions), dtype=np.float32
    )
    relevant_policies[np.arange(n_neighbors), best_action_indices] = 1.0

    probabilities = lambdas @ relevant_policies

    if not np.isclose(np.sum(probabilities), 1.0, atol=1e-2):
        raise ValueError(
            f"Interpolated probabilities do not sum to 1. "
            f"Sum: {np.sum(probabilities)}"
        )

    # Convex combination: weighted average of all actions by their probability mass
    optimal_action = probabilities @ optimal_policy.action_space

    return optimal_action, probabilities, None
