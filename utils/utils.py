import gymnasium as gym
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy.interpolate import griddata


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


@nb.njit(parallel=True, fastmath=True, cache=True)
def evaluate_policy_step(
    lambdas: np.ndarray,
    point_indexes: np.ndarray,
    value_function: np.ndarray,
    reward: np.ndarray,
    policy: np.ndarray,
    gamma: float,
    new_val: np.ndarray,
    terminal_mask: np.ndarray,
) -> None:
    """
    Execute zero-allocation in-place Bellman expectation update.

    Calculates the new values in parallel without triggering Numba reduction bugs.
    """
    n_states, n_actions, n_corners = lambdas.shape

    for s in nb.prange(n_states):
        if not terminal_mask[s]:
            v_s = 0.0
            for a in range(n_actions):
                if policy[s, a] > 0:
                    expected_next_v = 0.0
                    for k in range(n_corners):
                        idx = point_indexes[s, a, k]
                        expected_next_v += lambdas[s, a, k] * value_function[idx]

                    q_sa = reward[s, a] + gamma * expected_next_v
                    v_s += policy[s, a] * q_sa

            new_val[s] = v_s



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

# =====================================================================
# Visualization and Utility Functions
# =====================================================================


def plot_gradient_regions(
    bins_space: dict,
    states_space: np.ndarray,
    value_function: np.ndarray,
    quantile: float = 0.90,
    env=None,
    normalize: bool = False,
    show: bool = False,
    figsize: tuple = (12, 8),
) -> tuple:
    """Visualize regions of high and low gradients in a value function."""
    vs = value_function.copy()
    ss = states_space.copy()

    spacings = [space[1] - space[0] for space in bins_space.values()]
    shape = [len(space) for space in bins_space.values()]

    value_nd = vs.reshape(*shape)
    if normalize:
        val_range = value_nd.max() - value_nd.min() + 1e-8
        value_nd = (value_nd - value_nd.min()) / val_range

    grads = np.gradient(value_nd, *spacings, edge_order=2)
    grads_flat = np.stack([g.ravel() for g in grads], axis=1)
    grad_norm = np.linalg.norm(grads_flat, axis=1)

    thresh = np.quantile(grad_norm, quantile)
    high_mask = grad_norm > thresh
    low_mask = ~high_mask

    high_pts = ss[high_mask]
    low_pts = ss[low_mask]

    if show:
        plt.figure(figsize=figsize)
        plt.scatter(
            low_pts[:, 0],
            low_pts[:, 1],
            c="green",
            s=10,
            alpha=0.5,
            label=f"Low-Grad (bottom {100*(1-quantile):.0f}%)",
        )
        plt.scatter(
            high_pts[:, 0],
            high_pts[:, 1],
            c="red",
            s=30,
            marker="x",
            linewidths=1.5,
            label=f"High-Grad (top {100*quantile:.0f}%)",
        )

        dim_names = list(bins_space.keys())
        plt.xlabel(dim_names[0], fontsize=12)
        plt.ylabel(dim_names[1], fontsize=12)
        plt.title("State Space: High vs Low Gradient Regions", fontsize=14)
        plt.legend(loc="lower right")

        if env is not None:
            plt.xlim(
                getattr(env, f"min_{dim_names[0]}", states_space[:, 0].min()),
                getattr(env, f"max_{dim_names[0]}", states_space[:, 0].max()),
            )
            plt.ylim(
                getattr(env, f"min_{dim_names[1]}", states_space[:, 1].min()),
                getattr(env, f"max_{dim_names[1]}", states_space[:, 1].max()),
            )

        plt.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.show()

    return high_pts, low_pts, grad_norm, thresh


def plot_3D_value_function(
    vf: np.ndarray,
    points: np.ndarray,
    normalize: bool = True,
    cmap: str = "turbo_r",
    show: bool = False,
    path: str = "",
) -> None:
    """Plot a 3D value function in a color scale."""
    X = points[:, 0]
    Y = points[:, 1]
    vf_to_plot = (vf - vf.min()) / (vf.max() - vf.min() + 1e-8) if normalize else vf

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(X, Y, vf_to_plot, cmap=cmap, edgecolor="white", linewidth=0.2)

    vals_x = np.round(np.linspace(min(X), max(X), 4), 2)
    vals_y = np.round(np.linspace(min(Y), max(Y), 4), 2)
    ax.set_xticks(vals_x)
    ax.set_yticks(vals_y)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Value Function")

    if path:
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def plot_2D_value_function_with_cuts(
    pi: any,
    bins_space: dict,
    x_bin_name: str,
    y_bin_name: str,
    fixed_values: dict = None,
    cmap: str = "hot_r",
    levels: int = 50,
    method: str = "nearest",
) -> None:
    """Plot a 2D cross-section of an N-dimensional value function."""
    state_dim_names = list(bins_space.keys())
    num_dimensions = len(state_dim_names)

    if x_bin_name not in bins_space or y_bin_name not in bins_space:
        raise ValueError(f"Invalid bin names! Available: {state_dim_names}")

    x_vals = bins_space[x_bin_name]
    y_vals = bins_space[y_bin_name]
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")

    if num_dimensions == 2:
        query_points = np.column_stack([X.ravel(), Y.ravel()])
    else:
        if fixed_values is None:
            raise ValueError("For >2D state spaces, fixed_values must be provided.")

        grid_points = []
        for dim in state_dim_names:
            if dim == x_bin_name:
                grid_points.append(X)
            elif dim == y_bin_name:
                grid_points.append(Y)
            elif dim in fixed_values:
                grid_points.append(np.full_like(X, fixed_values[dim]))
            else:
                raise ValueError(f"No value specified for dimension '{dim}'")

        query_points = np.stack(grid_points, axis=-1).reshape(-1, num_dimensions)

    Z_flat = griddata(pi.states_space, pi.value_function, query_points, method=method)
    Z = Z_flat.reshape(X.shape)

    plt.figure(figsize=(8, 6))
    contourf = plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
    cbar = plt.colorbar(contourf)
    cbar.set_label("Value Function")
    plt.contour(X, Y, Z, levels=[0], colors="black", linewidths=1.0, linestyles="--")

    plt.xlabel(x_bin_name.replace("_", " ").title())
    plt.ylabel(y_bin_name.replace("_", " ").title())

    if fixed_values:
        fixed_txt = ", ".join([f"{k}={v}" for k, v in fixed_values.items()])
        plt.title(f"{x_bin_name} vs {y_bin_name} with {fixed_txt}")
    else:
        plt.title(f"{x_bin_name} vs {y_bin_name}")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def test_environment(
    task: gym.Env,
    pi: any,
    num_episodes: int = 10000,
    episode_length: int = 1000,
    option_reset: dict = None,
) -> None:
    """Test the environment using the provided policy."""
    for episode in range(num_episodes):
        total_reward = 0
        observation, _ = task.reset(options=option_reset)

        for timestep in range(1, episode_length):
            action, _ = get_optimal_action(observation, pi)
            observation, reward, terminated, _, _ = task.step_to_render(action)
            total_reward += reward

            if terminated or timestep == episode_length - 1:
                print(
                    f"Episode {episode} finished after {timestep} timesteps | "
                    f"Total reward: {total_reward}"
                )
                break
