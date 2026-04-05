"""
PolicyIteration.py
------------------
GPU-accelerated Policy Iteration for the 2-DOF Symmetric Pullout with thrust.

State  : (γ, V/Vs)   — flight-path angle [rad], normalised airspeed [-]
Action : (C_L, δ_throttle)  — lift coefficient [-], throttle fraction [0,1]
Reward : dt * V_norm * sin(γ)   (negative = altitude loss)
Terminal: γ >= 0 (pulled out) | γ <= -π (catastrophic dive)

All dynamics and 2-D barycentric interpolation run inside CUDA registers;
no transition-table is stored.  VRAM footprint is O(N_states).
"""

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
from loguru import logger

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

# ---------------------------------------------------------------------------
# Fused reduction: max |A - B|  (zero auxiliary VRAM)
# ---------------------------------------------------------------------------
max_abs_diff_kernel = cp.ReductionKernel(
    in_params="float32 x, float32 y",
    out_params="float32 z",
    map_expr="abs(x - y)",
    reduce_expr="max(a, b)",
    post_map_expr="z = a",
    identity="0.0f",
    name="max_abs_diff_2dof",
)


@dataclass
class PolicyIterationConfig:
    """Hyper-parameters for the Policy Iteration loop."""

    maximum_iterations: int = 20_000
    gamma: float = 1.0          # undiscounted
    theta: float = 1e-4         # convergence threshold
    n_steps: int = 100          # outer PI iterations
    log: bool = True
    log_interval: int = 150
    results_path: Path = field(default_factory=lambda: Path("results"))


class PolicyIteration:
    """
    Procedural Policy Iteration — 2-DOF Symmetric Pullout with Thrust.

    RK4 physics with thrust term and 2-D bilinear (barycentric) interpolation
    are embedded directly in CUDA C++ kernels; no lookup table is materialised
    in VRAM.

    action_space : (n_actions, 2) array of (CL, δ_throttle) pairs.
    """

    def __init__(
        self,
        env: gym.Env,
        states_space: np.ndarray,
        action_space: np.ndarray,
        config: PolicyIterationConfig,
    ) -> None:
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for CUDA kernels.")

        self.env = env
        self.states_space = np.ascontiguousarray(states_space, dtype=np.float32)
        # action_space: (n_actions, 2)  —  columns: [CL, δ_throttle]
        self.action_space = np.ascontiguousarray(action_space, dtype=np.float32)
        self.config = config

        self.n_states, self.n_dims = self.states_space.shape   # n_dims == 2
        self.n_actions = len(self.action_space)                # n_CL * n_throttle

        # Stall speed consistent with grumman.py constants
        cl_ref = 0.41 + 4.6983 * np.deg2rad(15)
        self.v_stall = float(
            np.sqrt(697.18 * 9.81 / (0.5 * 1.225 * 9.1147 * cl_ref))
        )
        # Throttle mapping: Kt  [N]  — full throttle = drag at Vmax = 2*Vs
        self.throttle_mapping = float(
            env.unwrapped.airplane.THROTTLE_LINEAR_MAPPING
        )
        logger.info(f"V_stall = {self.v_stall:.3f} m/s")
        logger.info(f"THROTTLE_LINEAR_MAPPING = {self.throttle_mapping:.2f} N")

        self._precompute_grid_metadata()
        self._allocate_tensors_and_compile()

    # ------------------------------------------------------------------
    # Grid metadata
    # ------------------------------------------------------------------

    def _precompute_grid_metadata(self) -> None:
        self.bounds_low = np.min(self.states_space, axis=0).astype(np.float32)
        self.bounds_high = np.max(self.states_space, axis=0).astype(np.float32)

        self.grid_shape = np.array(
            [len(np.unique(self.states_space[:, d])) for d in range(self.n_dims)],
            dtype=np.int32,
        )

        self.strides = np.zeros(self.n_dims, dtype=np.int32)
        stride = 1
        for d in range(self.n_dims - 1, -1, -1):
            self.strides[d] = stride
            stride *= self.grid_shape[d]

        logger.info(
            f"Grid: shape={self.grid_shape.tolist()}, "
            f"strides={self.strides.tolist()}, "
            f"N={self.n_states}"
        )

    # ------------------------------------------------------------------
    # GPU tensor allocation + kernel compilation
    # ------------------------------------------------------------------

    def _allocate_tensors_and_compile(self) -> None:
        logger.info("Allocating VRAM and compiling CUDA kernels …")

        self.d_states = cp.asarray(self.states_space, dtype=cp.float32)
        # actions stored flat: a_idx -> (actions[a_idx*2+0], actions[a_idx*2+1])
        self.d_actions = cp.asarray(self.action_space.ravel(), dtype=cp.float32)
        self.d_bounds_low = cp.asarray(self.bounds_low, dtype=cp.float32)
        self.d_bounds_high = cp.asarray(self.bounds_high, dtype=cp.float32)
        self.d_grid_shape = cp.asarray(self.grid_shape, dtype=cp.int32)
        self.d_strides = cp.asarray(self.strides, dtype=cp.int32)

        self.d_policy = cp.zeros(self.n_states, dtype=cp.int32)
        self.d_value_function = cp.zeros(self.n_states, dtype=cp.float32)
        self.d_new_value_function = cp.zeros(self.n_states, dtype=cp.float32)

        # Terminal mask (computed on CPU, pushed to GPU)
        terminal_mask, terminal_rewards = self.env.terminal(self.states_space)
        self.d_terminal_mask = cp.asarray(terminal_mask, dtype=cp.bool_)

        if np.isscalar(terminal_rewards):
            self.d_value_function[self.d_terminal_mask] = terminal_rewards
        else:
            self.d_value_function[self.d_terminal_mask] = cp.asarray(
                terminal_rewards[terminal_mask], dtype=cp.float32
            )
        self.d_new_value_function[:] = self.d_value_function[:]

        self._compile_cuda_module()
        logger.success("CUDA kernels compiled.")

    def _compile_cuda_module(self) -> None:
        cuda_source = r"""
        extern "C" {

        // ----------------------------------------------------------------
        // 2-DOF dynamics with thrust: returns d_gamma, d_vn
        // ----------------------------------------------------------------
        __device__ void get_derivatives_2dof_thrust(
            float gamma, float vn, float cl, float throttle,
            float& d_gamma, float& d_vn,
            float v_stall, float throttle_mapping
        ) {
            const float MASS  = 697.18f;
            const float S     = 9.1147f;
            const float RHO   = 1.225f;
            const float G     = 9.81f;
            const float CD0   = 0.0525f;
            const float CDA   = 0.2068f;
            const float CDA2  = 1.8712f;
            const float CL0   = 0.41f;
            const float CLA   = 4.6983f;

            float v_true  = vn * v_stall;
            float alpha   = (cl - CL0) / CLA;
            float cd      = CD0 + CDA * alpha + CDA2 * alpha * alpha;
            float dyn     = 0.5f * RHO * (S / MASS);
            float thrust  = throttle_mapping * throttle / MASS;   // a [m/s^2]

            // V_norm_dot = (V_dot) / v_stall
            d_vn = (-G * sinf(gamma)
                    - dyn * v_true * v_true * cd
                    + thrust) / v_stall;

            // gamma_dot  (mu = 0, cos(mu) = 1)
            float v_safe = fmaxf(v_true, 0.1f);
            d_gamma = dyn * v_true * cl - (G / v_safe) * cosf(gamma);
        }

        // ----------------------------------------------------------------
        // RK4 integrator - 2 states, returns step reward
        // ----------------------------------------------------------------
        __device__ void rk4_step_2dof_thrust(
            float& gamma, float& vn, float cl, float throttle,
            float dt, float v_stall, float throttle_mapping, float& reward
        ) {
            float k1g, k1v, k2g, k2v, k3g, k3v, k4g, k4v;

            get_derivatives_2dof_thrust(gamma,              vn,              cl, throttle, k1g, k1v, v_stall, throttle_mapping);
            get_derivatives_2dof_thrust(gamma+0.5f*dt*k1g, vn+0.5f*dt*k1v, cl, throttle, k2g, k2v, v_stall, throttle_mapping);
            get_derivatives_2dof_thrust(gamma+0.5f*dt*k2g, vn+0.5f*dt*k2v, cl, throttle, k3g, k3v, v_stall, throttle_mapping);
            get_derivatives_2dof_thrust(gamma+     dt*k3g, vn+     dt*k3v, cl, throttle, k4g, k4v, v_stall, throttle_mapping);

            gamma += (dt / 6.0f) * (k1g + 2.0f*k2g + 2.0f*k3g + k4g);
            vn    += (dt / 6.0f) * (k1v + 2.0f*k2v + 2.0f*k3v + k4v);

            float gamma_mid = gamma - (dt / 6.0f) * (k1g + 2.0f*k2g + 2.0f*k3g + k4g) * 0.5f;
            float vn_mid    = vn    - (dt / 6.0f) * (k1v + 2.0f*k2v + 2.0f*k3v + k4v) * 0.5f;
            reward = dt * vn_mid * sinf(gamma_mid);
        }

        // ----------------------------------------------------------------
        // 2-D bilinear (barycentric) interpolation - 4 corners
        // ----------------------------------------------------------------
        __device__ void get_barycentric_2d(
            float g, float v,
            const float* b_low, const float* b_high, const int* g_shape, const int* strides,
            int* idxs, float* wgts
        ) {
            float n0 = (g - b_low[0]) / (b_high[0] - b_low[0]) * (g_shape[0] - 1);
            float n1 = (v - b_low[1]) / (b_high[1] - b_low[1]) * (g_shape[1] - 1);

            n0 = fmaxf(0.0f, fminf(n0, (float)(g_shape[0] - 1)));
            n1 = fmaxf(0.0f, fminf(n1, (float)(g_shape[1] - 1)));

            int i0 = (int)n0;
            int i1 = (int)n1;
            if (i0 >= g_shape[0] - 1) i0 = g_shape[0] - 2;
            if (i1 >= g_shape[1] - 1) i1 = g_shape[1] - 2;

            float d0 = n0 - (float)i0;
            float d1 = n1 - (float)i1;

            idxs[0] = (i0  )*strides[0] + (i1  )*strides[1];
            idxs[1] = (i0  )*strides[0] + (i1+1)*strides[1];
            idxs[2] = (i0+1)*strides[0] + (i1  )*strides[1];
            idxs[3] = (i0+1)*strides[0] + (i1+1)*strides[1];

            wgts[0] = (1.0f - d0) * (1.0f - d1);
            wgts[1] = (1.0f - d0) *         d1;
            wgts[2] =         d0  * (1.0f - d1);
            wgts[3] =         d0  *         d1;
        }

        // ----------------------------------------------------------------
        // Policy evaluation kernel
        // actions layout: [cl_0, th_0, cl_1, th_1, ...] (stride-2 pairs)
        // ----------------------------------------------------------------
        __global__ void policy_eval_kernel(
            const float* states, const float* actions, const int* policy,
            const float* V, float* new_V, const bool* is_term,
            const float* b_low, const float* b_high,
            const int* g_shape, const int* strides,
            int n_states, float gamma_discount, float dt,
            float v_stall, float throttle_mapping
        ) {
            int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (s_idx >= n_states) return;

            if (is_term[s_idx]) {
                new_V[s_idx] = V[s_idx];
                return;
            }

            int a_idx    = policy[s_idx];
            float gamma  = states[s_idx * 2 + 0];
            float vn     = states[s_idx * 2 + 1];
            float cl     = actions[a_idx * 2 + 0];
            float throttle = actions[a_idx * 2 + 1];

            float reward;
            rk4_step_2dof_thrust(gamma, vn, cl, throttle, dt, v_stall, throttle_mapping, reward);

            int   idxs[4];
            float wgts[4];
            get_barycentric_2d(gamma, vn, b_low, b_high, g_shape, strides, idxs, wgts);

            float expected_v = 0.0f;
            for (int i = 0; i < 4; ++i)
                expected_v += wgts[i] * V[idxs[i]];

            new_V[s_idx] = reward + gamma_discount * expected_v;
        }

        // ----------------------------------------------------------------
        // Policy improvement kernel
        // ----------------------------------------------------------------
        __global__ void policy_improve_kernel(
            const float* states, const float* actions, int* policy,
            const float* V, const bool* is_term,
            const float* b_low, const float* b_high,
            const int* g_shape, const int* strides,
            int n_states, int n_actions, float gamma_discount, float dt,
            float v_stall, float throttle_mapping,
            int* policy_changes
        ) {
            int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (s_idx >= n_states) return;
            if (is_term[s_idx]) return;

            float init_gamma = states[s_idx * 2 + 0];
            float init_vn    = states[s_idx * 2 + 1];

            float max_q = -1e30f;
            int   best_a = 0;

            for (int a = 0; a < n_actions; ++a) {
                float gamma    = init_gamma;
                float vn       = init_vn;
                float cl       = actions[a * 2 + 0];
                float throttle = actions[a * 2 + 1];

                float reward;
                rk4_step_2dof_thrust(gamma, vn, cl, throttle, dt, v_stall, throttle_mapping, reward);

                int   idxs[4];
                float wgts[4];
                get_barycentric_2d(gamma, vn, b_low, b_high, g_shape, strides, idxs, wgts);

                float expected_v = 0.0f;
                for (int i = 0; i < 4; ++i)
                    expected_v += wgts[i] * V[idxs[i]];

                float q = reward + gamma_discount * expected_v;
                if (q > max_q) {
                    max_q  = q;
                    best_a = a;
                }
            }

            if (policy[s_idx] != best_a) {
                policy[s_idx] = best_a;
                atomicAdd(policy_changes, 1);
            }
        }

        } // extern "C"
        """

        module = cp.RawModule(code=cuda_source)
        self.eval_kernel    = module.get_function("policy_eval_kernel")
        self.improve_kernel = module.get_function("policy_improve_kernel")

        threads = 256
        self.threads_per_block = threads
        self.blocks_per_grid   = (self.n_states + threads - 1) // threads

    # ------------------------------------------------------------------
    # Policy Iteration phases
    # ------------------------------------------------------------------

    def policy_evaluation(self) -> float:
        delta        = float("inf")
        SYNC_INTERVAL = 25
        dt = np.float32(self.env.unwrapped.airplane.TIME_STEP)

        for i in range(self.config.maximum_iterations):
            self.eval_kernel(
                (self.blocks_per_grid,), (self.threads_per_block,),
                (
                    self.d_states, self.d_actions, self.d_policy,
                    self.d_value_function, self.d_new_value_function,
                    self.d_terminal_mask,
                    self.d_bounds_low, self.d_bounds_high,
                    self.d_grid_shape, self.d_strides,
                    np.int32(self.n_states),
                    np.float32(self.config.gamma),
                    dt,
                    np.float32(self.v_stall),
                    np.float32(self.throttle_mapping),
                ),
            )
            d_delta = max_abs_diff_kernel(self.d_new_value_function, self.d_value_function)
            self.d_value_function, self.d_new_value_function = (
                self.d_new_value_function,
                self.d_value_function,
            )

            if i % SYNC_INTERVAL == 0 or i == self.config.maximum_iterations - 1:
                delta = float(d_delta.get())
                if self.config.log and i % self.config.log_interval == 0:
                    logger.info(f"  eval step {i:>6d}  Δ={delta:.5e}")
                if delta < self.config.theta:
                    logger.success(f"Evaluation converged at step {i}  Δ={delta:.5e}")
                    return delta

        logger.warning(
            f"Evaluation hit max_iterations={self.config.maximum_iterations}  Δ={delta:.5e}"
        )
        return delta

    def policy_improvement(self) -> bool:
        d_changes = cp.zeros(1, dtype=cp.int32)
        dt = np.float32(self.env.unwrapped.airplane.TIME_STEP)

        self.improve_kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (
                self.d_states, self.d_actions, self.d_policy,
                self.d_value_function, self.d_terminal_mask,
                self.d_bounds_low, self.d_bounds_high,
                self.d_grid_shape, self.d_strides,
                np.int32(self.n_states),
                np.int32(self.n_actions),
                np.float32(self.config.gamma),
                dt,
                np.float32(self.v_stall),
                np.float32(self.throttle_mapping),
                d_changes,
            ),
        )

        changes = int(d_changes.get()[0])
        if changes:
            logger.info(f"Policy updated: {changes} states changed action.")
        return changes == 0

    def _pull_tensors_from_gpu(self) -> None:
        self.value_function = np.empty(self.n_states, dtype=np.float32)
        self.policy         = np.empty(self.n_states, dtype=np.int32)

        chunk = 5_000_000
        for i in range(0, self.n_states, chunk):
            end = min(i + chunk, self.n_states)
            self.d_value_function[i:end].get(out=self.value_function[i:end])
            self.d_policy[i:end].get(out=self.policy[i:end])

        del self.d_value_function, self.d_policy
        cp.get_default_memory_pool().free_all_blocks()
        logger.success("GPU tensors pulled to host RAM.")

    def run(self) -> None:
        for n in range(self.config.n_steps):
            logger.info(f"--- PI iteration {n + 1}/{self.config.n_steps} ---")
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                logger.success(f"Converged at PI iteration {n + 1}.")
                break

        self._pull_tensors_from_gpu()
        self.save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: Path | None = None) -> None:
        if filepath is None:
            filepath = Path.cwd() / f"{self.env.unwrapped.__class__.__name__}_policy.npz"
        filepath = filepath.with_suffix(".npz")
        np.savez(
            filepath,
            value_function=self.value_function,
            policy=self.policy,
            bounds_low=self.bounds_low,
            bounds_high=self.bounds_high,
            grid_shape=self.grid_shape,
            strides=self.strides,
            action_space=self.action_space,   # shape (n_actions, 2)
        )
        logger.success(f"Policy saved → {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: Path, env: gym.Env = None) -> "PolicyIteration":
        filepath = filepath.with_suffix(".npz")
        logger.info(f"Loading policy from {filepath.resolve()} …")
        data = np.load(filepath)

        inst = cls.__new__(cls)
        inst.env          = env
        inst.config       = PolicyIterationConfig()
        inst.value_function = data["value_function"]
        inst.policy         = data["policy"]
        inst.bounds_low     = data["bounds_low"]
        inst.bounds_high    = data["bounds_high"]
        inst.grid_shape     = data["grid_shape"]
        inst.strides        = data["strides"]
        inst.action_space   = data["action_space"]   # (n_actions, 2)
        inst.n_actions      = len(inst.action_space)
        inst.n_dims         = len(inst.bounds_low)
        inst.states_space   = None
        inst.n_states       = int(np.prod(inst.grid_shape))

        logger.success("Policy loaded.")
        return inst
