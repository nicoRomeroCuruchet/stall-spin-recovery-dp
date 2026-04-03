from dataclasses import dataclass, field
from itertools import product
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


# Fused Reduction Kernel: Computes max(abs(A - B)) with 0 bytes of auxiliary VRAM allocation
max_abs_diff_kernel = cp.ReductionKernel(
    in_params='float32 x, float32 y',
    out_params='float32 z',
    map_expr='abs(x - y)',
    reduce_expr='max(a, b)',
    post_map_expr='z = a',
    identity='0.0f',
    name='max_abs_diff_stall'
)


@dataclass
class PolicyIterationStallConfig:
    """
    Configuration parameters for the 4-DOF Symmetric Stall Policy Iteration.
    """
    maximum_iterations: int = 20_000
    gamma: float = 1.0
    theta: float = 1e-4
    n_steps: int = 100
    log: bool = False
    log_interval: int = 150
    img_path: Path = field(default_factory=lambda: Path("./img"))

    # --- Reward Shaping Weights (injected into CUDA kernel at compile time) ---
    w_q_penalty: float = 2.0           # Pitch damping: penalizes q² per timestep
    w_alpha_barrier_pos: float = 100.0  # Alpha barrier above positive stall
    w_alpha_barrier_neg: float = 10.0   # Alpha barrier below negative stall
    w_crash_penalty: float = 1000.0     # Crash penalty multiplier (×V_stall)
    w_control_effort: float = 1.0       # Control effort: penalizes δe²
    w_throttle_bonus: float = 0.2       # Throttle incentive bonus


class PolicyIterationStall:
    """
    High-performance Procedural Policy Iteration for 4-DOF Symmetric Stall Recovery.

    State space: (γ, V/Vs, α, q) — flight path angle, normalized airspeed,
                                     angle of attack, pitch rate.
    Action space: (δe, δt) — elevator deflection, throttle.

    Embeds the full Grumman AA-1 Yankee aerodynamic model (CL, CD, Cm with stall
    saturation) and 4D Barycentric interpolation directly into CUDA C++ kernels.
    """

    def __init__(
        self,
        env: gym.Env,
        states_space: np.ndarray,
        action_space: np.ndarray,
        config: PolicyIterationStallConfig,
    ) -> None:

        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for procedural on-the-fly CUDA kernels.")

        self.env = env
        self.states_space = np.ascontiguousarray(states_space, dtype=np.float32)
        self.action_space = np.ascontiguousarray(action_space, dtype=np.float32)
        self.config = config

        self.n_states, self.n_dims = self.states_space.shape
        self.n_actions = len(self.action_space)
        self.n_corners = 2**self.n_dims  # 16 for 4D

        # Aerodynamic constants from Grumman base class
        airplane = env.airplane
        self.v_stall = airplane.STALL_AIRSPEED
        self.k_thrust = airplane.THROTTLE_LINEAR_MAPPING
        self.dt = airplane.TIME_STEP

        self._precompute_grid_metadata()
        self._allocate_tensors_and_compile()

    def _precompute_grid_metadata(self) -> None:
        """Extract bounds, shape, and strides for the 4D CUDA interpolation."""
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

        self.corner_bits = np.array(
            list(product([0, 1], repeat=self.n_dims)), dtype=np.int32
        )
        logger.info(f"Grid precomputed. Shape: {self.grid_shape}, Strides: {self.strides}")

    def _allocate_tensors_and_compile(self) -> None:
        """Allocates minimal required memory and compiles the 4-DOF CUDA Kernels."""
        logger.info("Allocating procedural tensors and compiling 4-DOF CUDA JIT Kernels...")

        # Push constants to VRAM
        self.d_states = cp.asarray(self.states_space, dtype=cp.float32)
        self.d_actions = cp.asarray(self.action_space, dtype=cp.float32)
        self.d_bounds_low = cp.asarray(self.bounds_low, dtype=cp.float32)
        self.d_bounds_high = cp.asarray(self.bounds_high, dtype=cp.float32)
        self.d_grid_shape = cp.asarray(self.grid_shape, dtype=cp.int32)
        self.d_strides = cp.asarray(self.strides, dtype=cp.int32)

        # 1D Policy mapping state indices to the best action index
        self.d_policy = cp.zeros(self.n_states, dtype=cp.int32)

        self.d_value_function = cp.zeros(self.n_states, dtype=cp.float32)
        self.d_new_value_function = cp.zeros(self.n_states, dtype=cp.float32)

        # Compute terminal states in Python, push mask to GPU
        terminal_mask, terminal_rewards = self.env.terminal(self.states_space)
        self.d_terminal_mask = cp.asarray(terminal_mask, dtype=cp.bool_)

        if np.isscalar(terminal_rewards):
            self.d_value_function[self.d_terminal_mask] = terminal_rewards
        else:
            self.d_value_function[self.d_terminal_mask] = cp.asarray(
                terminal_rewards[terminal_mask], dtype=cp.float32
            )

        self.d_new_value_function[:] = self.d_value_function[:]

        # Compile the RawModule
        self._compile_cuda_module()
        logger.success("4-DOF CUDA Kernels compiled. VRAM usage optimized.")

    def _compile_cuda_module(self) -> None:
        """
        Compiles the 4-DOF aerodynamics and Bellman operations into CUDA.
        Implements a Markovian Alpha Barrier to enforce proper stall recovery physics.
        """
        reward_defines = f'''
        #define W_Q_PENALTY {self.config.w_q_penalty:.6f}f
        #define W_ALPHA_BARRIER_POS {self.config.w_alpha_barrier_pos:.6f}f
        #define W_ALPHA_BARRIER_NEG {self.config.w_alpha_barrier_neg:.6f}f
        #define W_CRASH_PENALTY {self.config.w_crash_penalty:.6f}f
        #define W_CONTROL_EFFORT {self.config.w_control_effort:.6f}f
        #define W_THROTTLE_BONUS {self.config.w_throttle_bonus:.6f}f
        '''
        cuda_source = reward_defines + r'''
        extern "C" {

        __device__ const float MASS = 697.18f;
        __device__ const float S = 9.1147f;
        __device__ const float CHORD = 1.22f;
        __device__ const float RHO = 1.225f;
        __device__ const float G = 9.81f;
        __device__ const float I_YY = 1011.43f;
        __device__ const float CL_0 = 0.41f;
        __device__ const float CL_ALPHA = 4.6983f;
        __device__ const float CL_DE = 0.361f;
        __device__ const float CL_QHAT = 2.42f;
        __device__ const float CD_0 = 0.0525f;
        __device__ const float CD_ALPHA = 0.2068f;
        __device__ const float CD_ALPHA2 = 1.8712f;
        __device__ const float CM_0 = 0.076f;
        __device__ const float CM_ALPHA = -0.8938f;
        __device__ const float CM_DE = -1.0313f;
        __device__ const float CM_QHAT = -7.15f;
        __device__ const float ALPHA_STALL_POS = 0.2618f;
        __device__ const float ALPHA_STALL_NEG = -0.12217f;
        __device__ const float CL_AT_POS_STALL = 0.41f + 4.6983f * 0.2618f;
        __device__ const float CL_AT_NEG_STALL = 0.41f + 4.6983f * (-0.12217f);

        __device__ void rk4_step_4dof(
            float& gamma, float& vn, float& alpha, float& q,
            float de, float throttle, float dt_micro, int n_micro,
            float v_stall, float k_thrust, float& total_reward
        ) {
            total_reward = 0.0f;
            float thrust_force = k_thrust * throttle;

            for(int m = 0; m < n_micro; ++m) {
                float k1_g, k1_v, k1_a, k1_q, k2_g, k2_v, k2_a, k2_q;
                float k3_g, k3_v, k3_a, k3_q, k4_g, k4_v, k4_a, k4_q;

                // --- k1 ---
                {
                    float vt = fmaxf(vn * v_stall, 0.1f);
                    float q_hat = q * CHORD / (2.0f * vt);
                    float cl, cd, cm;
                    if (alpha >= ALPHA_STALL_POS) cl = CL_AT_POS_STALL;
                    else if (alpha <= ALPHA_STALL_NEG) cl = CL_AT_NEG_STALL;
                    else cl = CL_0 + CL_ALPHA * alpha + CL_DE * de + CL_QHAT * q_hat;
                    cd = CD_0 + CD_ALPHA * alpha + CD_ALPHA2 * alpha * alpha;
                    cm = CM_0 + CM_ALPHA * alpha + CM_DE * de + CM_QHAT * q_hat;
                    float qS = 0.5f * RHO * S * vt * vt;
                    float L = qS * cl;
                    float D = qS * cd;
                    float My = qS * CHORD * cm;
                    k1_g = (L + thrust_force * sinf(alpha)) / (MASS * vt) - (G / vt) * cosf(gamma);
                    k1_v = (-G * sinf(gamma) - (D - thrust_force * cosf(alpha)) / MASS) / v_stall;
                    k1_a = q - k1_g;
                    k1_q = My / I_YY;
                }
                // --- k2 ---
                {
                    float g2 = gamma + 0.5f*dt_micro*k1_g, v2 = vn + 0.5f*dt_micro*k1_v;
                    float a2 = alpha + 0.5f*dt_micro*k1_a, q2 = q + 0.5f*dt_micro*k1_q;
                    float vt = fmaxf(v2 * v_stall, 0.1f);
                    float q_hat = q2 * CHORD / (2.0f * vt);
                    float cl, cd, cm;
                    if (a2 >= ALPHA_STALL_POS) cl = CL_AT_POS_STALL;
                    else if (a2 <= ALPHA_STALL_NEG) cl = CL_AT_NEG_STALL;
                    else cl = CL_0 + CL_ALPHA * a2 + CL_DE * de + CL_QHAT * q_hat;
                    cd = CD_0 + CD_ALPHA * a2 + CD_ALPHA2 * a2 * a2;
                    cm = CM_0 + CM_ALPHA * a2 + CM_DE * de + CM_QHAT * q_hat;
                    float qS = 0.5f * RHO * S * vt * vt;
                    float L = qS * cl;
                    float D = qS * cd;
                    float My = qS * CHORD * cm;
                    k2_g = (L + thrust_force * sinf(a2)) / (MASS * vt) - (G / vt) * cosf(g2);
                    k2_v = (-G * sinf(g2) - (D - thrust_force * cosf(a2)) / MASS) / v_stall;
                    k2_a = q2 - k2_g;
                    k2_q = My / I_YY;
                }
                // --- k3 ---
                {
                    float g3 = gamma + 0.5f*dt_micro*k2_g, v3 = vn + 0.5f*dt_micro*k2_v;
                    float a3 = alpha + 0.5f*dt_micro*k2_a, q3 = q + 0.5f*dt_micro*k2_q;
                    float vt = fmaxf(v3 * v_stall, 0.1f);
                    float q_hat = q3 * CHORD / (2.0f * vt);
                    float cl, cd, cm;
                    if (a3 >= ALPHA_STALL_POS) cl = CL_AT_POS_STALL;
                    else if (a3 <= ALPHA_STALL_NEG) cl = CL_AT_NEG_STALL;
                    else cl = CL_0 + CL_ALPHA * a3 + CL_DE * de + CL_QHAT * q_hat;
                    cd = CD_0 + CD_ALPHA * a3 + CD_ALPHA2 * a3 * a3;
                    cm = CM_0 + CM_ALPHA * a3 + CM_DE * de + CM_QHAT * q_hat;
                    float qS = 0.5f * RHO * S * vt * vt;
                    float L = qS * cl;
                    float D = qS * cd;
                    float My = qS * CHORD * cm;
                    k3_g = (L + thrust_force * sinf(a3)) / (MASS * vt) - (G / vt) * cosf(g3);
                    k3_v = (-G * sinf(g3) - (D - thrust_force * cosf(a3)) / MASS) / v_stall;
                    k3_a = q3 - k3_g;
                    k3_q = My / I_YY;
                }
                // --- k4 ---
                {
                    float g4 = gamma + dt_micro*k3_g, v4 = vn + dt_micro*k3_v;
                    float a4 = alpha + dt_micro*k3_a, q4 = q + dt_micro*k3_q;
                    float vt = fmaxf(v4 * v_stall, 0.1f);
                    float q_hat = q4 * CHORD / (2.0f * vt);
                    float cl, cd, cm;
                    if (a4 >= ALPHA_STALL_POS) cl = CL_AT_POS_STALL;
                    else if (a4 <= ALPHA_STALL_NEG) cl = CL_AT_NEG_STALL;
                    else cl = CL_0 + CL_ALPHA * a4 + CL_DE * de + CL_QHAT * q_hat;
                    cd = CD_0 + CD_ALPHA * a4 + CD_ALPHA2 * a4 * a4;
                    cm = CM_0 + CM_ALPHA * a4 + CM_DE * de + CM_QHAT * q_hat;
                    float qS = 0.5f * RHO * S * vt * vt;
                    float L = qS * cl;
                    float D = qS * cd;
                    float My = qS * CHORD * cm;
                    k4_g = (L + thrust_force * sinf(a4)) / (MASS * vt) - (G / vt) * cosf(g4);
                    k4_v = (-G * sinf(g4) - (D - thrust_force * cosf(a4)) / MASS) / v_stall;
                    k4_a = q4 - k4_g;
                    k4_q = My / I_YY;
                }

                gamma += (dt_micro / 6.0f) * (k1_g + 2.0f*k2_g + 2.0f*k3_g + k4_g);
                vn    += (dt_micro / 6.0f) * (k1_v + 2.0f*k2_v + 2.0f*k3_v + k4_v);
                alpha += (dt_micro / 6.0f) * (k1_a + 2.0f*k2_a + 2.0f*k3_a + k4_a);
                q     += (dt_micro / 6.0f) * (k1_q + 2.0f*k2_q + 2.0f*k3_q + k4_q);

                // PURE MARKOVIAN PENALTIES
                // 1. Primary Physical Cost: Altitude loss
                total_reward += dt_micro * vn * v_stall * sinf(gamma);

                if (gamma >= 0.0f) { break; }

                // 2. Pitch Damping Penalty
                total_reward -= W_Q_PENALTY * (q * q) * dt_micro;

                // 3. MARKOVIAN ALPHA BARRIER (Stall Prevention)
                if (alpha > 0.2618f) {
                    total_reward -= W_ALPHA_BARRIER_POS * (alpha - 0.2618f) * dt_micro;
                } else if (alpha < -0.12217f) {
                    total_reward -= W_ALPHA_BARRIER_NEG * (-alpha - 0.12217f) * dt_micro;
                }

                if (alpha >= 0.698132f || alpha <= -0.698132f || gamma <= -3.1f) {
                    total_reward -= W_CRASH_PENALTY * v_stall;
                    break;
                }
            }

            // 4. Control Effort Penalty
            total_reward -= W_CONTROL_EFFORT * (de * de) * (dt_micro * n_micro);
            total_reward += W_THROTTLE_BONUS * throttle * (dt_micro * n_micro);
        }

        __device__ void get_barycentric_4d(
            float s0, float s1, float s2, float s3,
            const float* b_low, const float* b_high, const int* g_shape, const int* strides,
            int* idxs, float* wgts
        ) {
            float n0 = (s0 - b_low[0]) / (b_high[0] - b_low[0]) * (g_shape[0] - 1);
            float n1 = (s1 - b_low[1]) / (b_high[1] - b_low[1]) * (g_shape[1] - 1);
            float n2 = (s2 - b_low[2]) / (b_high[2] - b_low[2]) * (g_shape[2] - 1);
            float n3 = (s3 - b_low[3]) / (b_high[3] - b_low[3]) * (g_shape[3] - 1);

            n0 = fmaxf(0.0f, fminf(n0, (float)(g_shape[0] - 1)));
            n1 = fmaxf(0.0f, fminf(n1, (float)(g_shape[1] - 1)));
            n2 = fmaxf(0.0f, fminf(n2, (float)(g_shape[2] - 1)));
            n3 = fmaxf(0.0f, fminf(n3, (float)(g_shape[3] - 1)));

            int i0 = (int)n0; int i1 = (int)n1; int i2 = (int)n2; int i3 = (int)n3;

            if (i0 == g_shape[0] - 1) i0--;
            if (i1 == g_shape[1] - 1) i1--;
            if (i2 == g_shape[2] - 1) i2--;
            if (i3 == g_shape[3] - 1) i3--;

            float d0 = n0 - i0; float d1 = n1 - i1;
            float d2 = n2 - i2; float d3 = n3 - i3;

            #pragma unroll
            for (int a = 0; a < 2; ++a) {
                #pragma unroll
                for (int b = 0; b < 2; ++b) {
                    #pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        #pragma unroll
                        for (int d = 0; d < 2; ++d) {
                            int corner = a * 8 + b * 4 + c * 2 + d;
                            idxs[corner] = (i0 + a) * strides[0] + (i1 + b) * strides[1]
                                         + (i2 + c) * strides[2] + (i3 + d) * strides[3];
                            wgts[corner] = (a ? d0 : (1.0f - d0)) * (b ? d1 : (1.0f - d1)) * (c ? d2 : (1.0f - d2)) * (d ? d3 : (1.0f - d3));
                        }
                    }
                }
            }
        }

        __global__ void policy_eval_kernel(
            const float* states, const float* actions, const int* policy,
            const float* V, float* new_V, const bool* is_term,
            const float* b_low, const float* b_high, const int* g_shape, const int* strides,
            int n_states, float gamma_discount, float dt, float v_stall, float k_thrust
        ) {
            int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (s_idx >= n_states) return;
            if (is_term[s_idx]) { new_V[s_idx] = V[s_idx]; return; }

            int a_idx = policy[s_idx];
            float gamma = states[s_idx * 4 + 0], vn = states[s_idx * 4 + 1];
            float alpha = states[s_idx * 4 + 2], q = states[s_idx * 4 + 3];
            float de = actions[a_idx * 2 + 0], throttle = actions[a_idx * 2 + 1];
            float reward;

            rk4_step_4dof(gamma, vn, alpha, q, de, throttle, dt, 1, v_stall, k_thrust, reward);

            int idxs[16]; float wgts[16];
            get_barycentric_4d(gamma, vn, alpha, q, b_low, b_high, g_shape, strides, idxs, wgts);

            float expected_v = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) { expected_v = fmaf(wgts[i], V[idxs[i]], expected_v); }
            new_V[s_idx] = reward + gamma_discount * expected_v;
        }

        __global__ void policy_improve_kernel(
            const float* states, const float* actions, int* policy,
            const float* V, const bool* is_term,
            const float* b_low, const float* b_high, const int* g_shape, const int* strides,
            int n_states, int n_actions, float gamma_discount, float dt,
            float v_stall, float k_thrust, int* policy_changes
        ) {
            int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (s_idx >= n_states) return;
            if (is_term[s_idx]) return;

            float init_gamma = states[s_idx * 4 + 0], init_vn = states[s_idx * 4 + 1];
            float init_alpha = states[s_idx * 4 + 2], init_q = states[s_idx * 4 + 3];
            float max_q_val = -1e9f; int best_a = 0;

            for (int a = 0; a < n_actions; ++a) {
                float gamma = init_gamma, vn = init_vn, alpha = init_alpha, q = init_q;
                float de = actions[a * 2 + 0], throttle = actions[a * 2 + 1], reward;

                rk4_step_4dof(gamma, vn, alpha, q, de, throttle, dt, 1, v_stall, k_thrust, reward);

                int idxs[16]; float wgts[16];
                get_barycentric_4d(gamma, vn, alpha, q, b_low, b_high, g_shape, strides, idxs, wgts);

                float expected_v = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) { expected_v = fmaf(wgts[i], V[idxs[i]], expected_v); }

                float q_val = reward + gamma_discount * expected_v;
                if (q_val > max_q_val) { max_q_val = q_val; best_a = a; }
            }

            if (policy[s_idx] != best_a) {
                policy[s_idx] = best_a;
                atomicAdd(policy_changes, 1);
            }
        }
        }
        '''
        module = cp.RawModule(code=cuda_source)
        self.eval_kernel = module.get_function('policy_eval_kernel')
        self.improve_kernel = module.get_function('policy_improve_kernel')
        self.threads_per_block = 256
        self.blocks_per_grid = (self.n_states + self.threads_per_block - 1) // self.threads_per_block

    def _pull_tensors_from_gpu(self) -> None:
        """
        Retrieves converged policy to CPU RAM using zero-copy transfers.
        """
        logger.info("Retrieving converged matrices from VRAM to CPU RAM...")

        gpu_tensors_to_free = [
            'd_new_value_function', 'd_terminal_mask',
            'd_states', 'd_actions',
            'd_bounds_low', 'd_bounds_high',
            'd_grid_shape', 'd_strides',
        ]
        for attr in gpu_tensors_to_free:
            if hasattr(self, attr):
                delattr(self, attr)

        cp.get_default_memory_pool().free_all_blocks()
        logger.info("Released unused GPU tensors and freed VRAM pool.")

        self.value_function = np.empty(self.n_states, dtype=np.float32)
        self.policy = np.empty(self.n_states, dtype=np.int32)

        chunk_size = 5_000_000
        for i in range(0, self.n_states, chunk_size):
            end = min(i + chunk_size, self.n_states)
            self.d_value_function[i:end].get(out=self.value_function[i:end])
            self.d_policy[i:end].get(out=self.policy[i:end])

        del self.d_value_function, self.d_policy
        cp.get_default_memory_pool().free_all_blocks()

        logger.success("Matrices successfully pulled to Host RAM. All VRAM released.")

    def policy_evaluation(self) -> float:
        """Executes purely procedural evaluation on GPU."""
        delta = float("inf")
        SYNC_INTERVAL = 25

        for i in range(self.config.maximum_iterations):
            self.eval_kernel(
                (self.blocks_per_grid,), (self.threads_per_block,),
                (
                    self.d_states, self.d_actions, self.d_policy,
                    self.d_value_function, self.d_new_value_function, self.d_terminal_mask,
                    self.d_bounds_low, self.d_bounds_high, self.d_grid_shape, self.d_strides,
                    np.int32(self.n_states), np.float32(self.config.gamma),
                    np.float32(self.dt), np.float32(self.v_stall), np.float32(self.k_thrust)
                )
            )

            d_delta = max_abs_diff_kernel(self.d_new_value_function, self.d_value_function)
            self.d_value_function, self.d_new_value_function = self.d_new_value_function, self.d_value_function

            if i % SYNC_INTERVAL == 0 or i == self.config.maximum_iterations - 1:
                delta = float(d_delta.get())
                if delta < self.config.theta:
                    logger.success(f"GPU Evaluation converged at step {i} with Δ={delta:.5e}")
                    return delta

        msg = f"GPU Evaluation hit max iterations ({self.config.maximum_iterations})"
        logger.warning(f"{msg} with Δ={delta:.5e}")
        return delta

    def policy_improvement(self) -> bool:
        """Executes procedural policy greedy improvement on GPU."""
        d_policy_changes = cp.zeros(1, dtype=cp.int32)

        self.improve_kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (
                self.d_states, self.d_actions, self.d_policy,
                self.d_value_function, self.d_terminal_mask,
                self.d_bounds_low, self.d_bounds_high, self.d_grid_shape, self.d_strides,
                np.int32(self.n_states), np.int32(self.n_actions),
                np.float32(self.config.gamma), np.float32(self.env.airplane.TIME_STEP),
                np.float32(self.env.airplane.STALL_AIRSPEED),
                np.float32(self.env.airplane.THROTTLE_LINEAR_MAPPING),
                d_policy_changes
            )
        )

        changes = int(d_policy_changes.get()[0])

        # FIX: Action Chattering Tolerance
        # En una grilla de 8.2M estados, permitimos que un ~0.01% (aprox 820 estados)
        # oscile por ruido de punto flotante entre acciones adyacentes.
        tolerance_threshold = int(self.n_states * 0.0001)

        policy_stable = (changes <= tolerance_threshold)

        if not policy_stable:
            logger.info(f"GPU Policy updated: {changes} states changed optimal action. (Tolerance: {tolerance_threshold})")

        return policy_stable

    def run(self) -> None:
        """Execute the complete Policy Iteration architecture."""
        for n in range(self.config.n_steps):
            logger.info(f"--- Iteration {n + 1}/{self.config.n_steps} ---")
            self.policy_evaluation()
            is_stable = self.policy_improvement()

            if is_stable:
                logger.success(f"Algorithm converged optimally at iteration {n + 1}.")
                break

        self._pull_tensors_from_gpu()
        self.save()

    def save(self, filepath: Path | None = None) -> None:
        """Serialize and save the trained model to disk."""
        if filepath is None:
            filepath = Path.cwd() / f"{self.env.unwrapped.__class__.__name__}_policy.npz"

        filepath = filepath.with_suffix(".npz")

        logger.info(f"Serializing policy to {filepath.resolve()}...")

        np.savez(
            filepath,
            value_function=self.value_function,
            policy=self.policy,
            bounds_low=self.bounds_low,
            bounds_high=self.bounds_high,
            grid_shape=self.grid_shape,
            strides=self.strides,
            corner_bits=self.corner_bits,
            action_space=self.action_space
        )

        logger.success(f"Policy saved successfully to {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: Path, env: gym.Env = None) -> "PolicyIterationStall":
        """Load a saved policy instance from a serialized .npz archive."""
        filepath = filepath.with_suffix(".npz")

        logger.info(f"Loading policy from {filepath.resolve()}...")
        data = np.load(filepath)

        instance = cls.__new__(cls)
        instance.env = env
        instance.config = PolicyIterationStallConfig()

        instance.value_function = data["value_function"]
        instance.policy = data["policy"]
        instance.bounds_low = data["bounds_low"]
        instance.bounds_high = data["bounds_high"]
        instance.grid_shape = data["grid_shape"]
        instance.strides = data["strides"]
        instance.corner_bits = data["corner_bits"]
        instance.action_space = data["action_space"]

        instance.n_actions = len(instance.action_space)
        instance.n_dims = len(instance.bounds_low)

        instance.states_space = None

        logger.success(f"Policy loaded successfully from {filepath.resolve()}")
        return instance
