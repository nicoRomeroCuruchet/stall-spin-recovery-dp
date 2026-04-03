import pickle
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

        // Riley (1985) NASA-TM-86309 Table III - alpha breakpoints (radians)
        __device__ const float CL_A_TBL[14] = {
            -0.17453f, -0.08727f,  0.00000f,  0.08727f,  0.17453f,
             0.20944f,  0.24435f,  0.27925f,  0.31416f,  0.34907f,
             0.43633f,  0.52360f,  0.61087f,  0.69813f
        };

        __device__ float cl_interp(float alpha, const float* tbl) {
            if (alpha <= CL_A_TBL[0])  return tbl[0];
            if (alpha >= CL_A_TBL[13]) return tbl[13];
            for (int i = 0; i < 13; ++i) {
                if (alpha <= CL_A_TBL[i+1]) {
                    float t = (alpha - CL_A_TBL[i]) / (CL_A_TBL[i+1] - CL_A_TBL[i]);
                    return tbl[i] + t * (tbl[i+1] - tbl[i]);
                }
            }
            return tbl[13];
        }

        __device__ float bilinear_interp(float alpha, float ct,
                                          const float* tbl0, const float* tbl05) {
            float t = fminf(fmaxf(ct / 0.5f, 0.0f), 1.0f);
            return cl_interp(alpha, tbl0) + t * (cl_interp(alpha, tbl05) - cl_interp(alpha, tbl0));
        }

        __device__ float compute_ct(float throttle, float vt, float k_thrust) {
            float q_bar = 0.5f * RHO * fmaxf(vt, 0.1f) * fmaxf(vt, 0.1f);
            return fminf(fmaxf(k_thrust * throttle / (q_bar * S), 0.0f), 0.5f);
        }

        // --- CT=0 tables (Riley 1985, Table III, power-off) ---

        // CL_o: flat-top 14-18 deg (max=1.26), gradual post-stall drop
        __device__ const float CL_O_TBL_CT0[14] = {
            -0.41f, -0.01f,  0.41f,  0.84f,  1.16f,
             1.23f,  1.26f,  1.26f,  1.26f,  1.25f,
             1.22f,  1.17f,  1.13f,  1.08f
        };

        // CL_q: pitch damping, CT=0
        __device__ const float CL_Q_TBL_CT0[14] = {
             2.41f,  2.41f,  2.42f,  2.46f,  2.59f,
             2.96f,  3.72f,  4.73f,  5.29f,  5.16f,
             5.05f,  5.06f,  5.98f,  5.08f
        };

        // CM_q: pitch damping, CT=0
        __device__ const float CM_Q_TBL_CT0[14] = {
             -7.0000f,  -7.0000f,  -7.0400f,  -7.1500f,  -7.5200f,
             -8.6200f, -10.8000f, -13.7300f, -15.3800f, -15.0000f,
            -14.6600f, -14.7100f, -14.7700f, -14.7700f
        };

        // CM_o: nonlinear nose-down moment, CT=0
        __device__ const float CM_O_TBL_CT0[14] = {
             0.2700f,  0.1580f,  0.0760f,  0.0020f, -0.0800f,
            -0.1180f, -0.1670f, -0.2250f, -0.2770f, -0.3160f,
            -0.4080f, -0.4800f, -0.5560f, -0.6060f
        };

        // CD_o: strong post-stall rise, CT=0
        __device__ const float CD_O_TBL_CT0[14] = {
            0.0666f, 0.0486f, 0.0526f, 0.0846f, 0.1456f,
            0.1856f, 0.2446f, 0.3136f, 0.3786f, 0.4486f,
            0.6186f, 0.7786f, 0.9255f, 1.0636f
        };

        // CL_de (elevator effectiveness), CT=0, /rad
        __device__ const float CL_DE_TBL_CT0[14] = {
            0.35523f, 0.36096f, 0.35523f, 0.33232f, 0.30367f,
            0.29221f, 0.28648f, 0.28075f, 0.27502f, 0.26929f,
            0.25210f, 0.24064f, 0.22345f, 0.21192f
        };

        // CM_de (elevator effectiveness), CT=0, /rad
        __device__ const float CM_DE_TBL_CT0[14] = {
            -1.10581f, -1.10581f, -1.10581f, -1.03132f, -0.94538f,
            -0.93965f, -0.93392f, -0.92819f, -0.92819f, -0.92819f,
            -0.92819f, -0.85944f, -0.74485f, -0.57296f
        };

        // --- CT=0.5 tables (Riley 1985, Table III, power-on, thrust embedded) ---

        // CL_o at CT=0.5: max 1.72 at 18 deg
        __device__ const float CL_O_TBL_CT05[14] = {
            -0.67f, -0.14f,  0.41f,  0.97f,  1.42f,
             1.54f,  1.62f,  1.67f,  1.72f,  1.76f,
             1.85f,  1.92f,  1.99f,  2.05f
        };

        // CL_q at CT=0.5
        __device__ const float CL_Q_TBL_CT05[14] = {
             3.012f,  3.012f,  3.029f,  3.222f,  3.594f,
             4.351f,  6.072f,  6.382f,  6.988f,  6.833f,
             6.561f,  6.127f,  5.966f,  5.811f
        };

        // CD_o at CT=0.5: negative at low alpha (propulsor generates more forward force than drag)
        __device__ const float CD_O_TBL_CT05[14] = {
            -0.3273f, -0.3494f, -0.3474f, -0.3139f, -0.2483f,
            -0.2057f, -0.1435f, -0.0709f, -0.0018f,  0.0727f,
             0.2561f,  0.4322f,  0.5979f,  0.7572f
        };

        // CM_o at CT=0.5
        __device__ const float CM_O_TBL_CT05[14] = {
             0.2700f,  0.1580f,  0.0760f,  0.0020f, -0.0800f,
            -0.1180f, -0.1670f, -0.2250f, -0.2770f, -0.3160f,
            -0.4080f, -0.4800f, -0.5560f, -0.6060f
        };

        // CM_q at CT=0.5
        __device__ const float CM_Q_TBL_CT05[14] = {
             -8.75f,  -8.75f,  -8.80f,  -9.36f, -10.44f,
            -12.64f, -17.64f, -18.54f, -20.30f, -19.85f,
            -19.06f, -17.80f, -17.33f, -16.88f
        };

        // CL_de at CT=0.5, /rad
        __device__ const float CL_DE_TBL_CT05[14] = {
            0.79641f, 0.77922f, 0.75018f, 0.70474f, 0.62453f,
            0.59588f, 0.57869f, 0.56150f, 0.53858f, 0.51566f,
            0.45837f, 0.41826f, 0.34950f, 0.31513f
        };

        // CM_de at CT=0.5, /rad
        __device__ const float CM_DE_TBL_CT05[14] = {
            -2.14484f, -2.25173f, -2.25746f, -2.26319f, -2.20016f,
            -2.06265f, -1.91348f, -1.78190f, -1.65032f, -1.54146f,
            -1.29488f, -1.22040f, -1.08862f, -0.85944f
        };

        __device__ void rk4_step_4dof(
            float& gamma, float& vn, float& alpha, float& q,
            float de, float throttle, float dt_micro, int n_micro,
            float v_stall, float k_thrust, float& total_reward
        ) {
            total_reward = 0.0f;

            for(int m = 0; m < n_micro; ++m) {
                float k1_g, k1_v, k1_a, k1_q, k2_g, k2_v, k2_a, k2_q;
                float k3_g, k3_v, k3_a, k3_q, k4_g, k4_v, k4_a, k4_q;

                // --- k1 ---
                {
                    float vt = fmaxf(vn * v_stall, 0.1f);
                    float q_hat = q * CHORD / (2.0f * vt);
                    float ct = compute_ct(throttle, vt, k_thrust);
                    float cl_de = bilinear_interp(alpha, ct, CL_DE_TBL_CT0, CL_DE_TBL_CT05);
                    float cm_de = bilinear_interp(alpha, ct, CM_DE_TBL_CT0, CM_DE_TBL_CT05);
                    float cl = bilinear_interp(alpha, ct, CL_O_TBL_CT0, CL_O_TBL_CT05)
                               + cl_de * de
                               + bilinear_interp(alpha, ct, CL_Q_TBL_CT0, CL_Q_TBL_CT05) * q_hat;
                    float cd = bilinear_interp(alpha, ct, CD_O_TBL_CT0, CD_O_TBL_CT05);
                    float cm = bilinear_interp(alpha, ct, CM_O_TBL_CT0, CM_O_TBL_CT05)
                               + cm_de * de
                               + bilinear_interp(alpha, ct, CM_Q_TBL_CT0, CM_Q_TBL_CT05) * q_hat;
                    float qS = 0.5f * RHO * S * vt * vt;
                    float L = qS * cl;
                    float D = qS * cd;
                    float My = qS * CHORD * cm;
                    k1_g = L / (MASS * vt) - (G / vt) * cosf(gamma);
                    k1_v = (-G * sinf(gamma) - D / MASS) / v_stall;
                    k1_a = q - k1_g;
                    k1_q = My / I_YY;
                }
                // --- k2 ---
                {
                    float g2 = gamma + 0.5f*dt_micro*k1_g, v2 = vn + 0.5f*dt_micro*k1_v;
                    float a2 = alpha + 0.5f*dt_micro*k1_a, q2 = q + 0.5f*dt_micro*k1_q;
                    float vt = fmaxf(v2 * v_stall, 0.1f);
                    float q_hat = q2 * CHORD / (2.0f * vt);
                    float ct = compute_ct(throttle, vt, k_thrust);
                    float cl_de = bilinear_interp(a2, ct, CL_DE_TBL_CT0, CL_DE_TBL_CT05);
                    float cm_de = bilinear_interp(a2, ct, CM_DE_TBL_CT0, CM_DE_TBL_CT05);
                    float cl = bilinear_interp(a2, ct, CL_O_TBL_CT0, CL_O_TBL_CT05)
                               + cl_de * de
                               + bilinear_interp(a2, ct, CL_Q_TBL_CT0, CL_Q_TBL_CT05) * q_hat;
                    float cd = bilinear_interp(a2, ct, CD_O_TBL_CT0, CD_O_TBL_CT05);
                    float cm = bilinear_interp(a2, ct, CM_O_TBL_CT0, CM_O_TBL_CT05)
                               + cm_de * de
                               + bilinear_interp(a2, ct, CM_Q_TBL_CT0, CM_Q_TBL_CT05) * q_hat;
                    float qS = 0.5f * RHO * S * vt * vt;
                    float L = qS * cl;
                    float D = qS * cd;
                    float My = qS * CHORD * cm;
                    k2_g = L / (MASS * vt) - (G / vt) * cosf(g2);
                    k2_v = (-G * sinf(g2) - D / MASS) / v_stall;
                    k2_a = q2 - k2_g;
                    k2_q = My / I_YY;
                }
                // --- k3 ---
                {
                    float g3 = gamma + 0.5f*dt_micro*k2_g, v3 = vn + 0.5f*dt_micro*k2_v;
                    float a3 = alpha + 0.5f*dt_micro*k2_a, q3 = q + 0.5f*dt_micro*k2_q;
                    float vt = fmaxf(v3 * v_stall, 0.1f);
                    float q_hat = q3 * CHORD / (2.0f * vt);
                    float ct = compute_ct(throttle, vt, k_thrust);
                    float cl_de = bilinear_interp(a3, ct, CL_DE_TBL_CT0, CL_DE_TBL_CT05);
                    float cm_de = bilinear_interp(a3, ct, CM_DE_TBL_CT0, CM_DE_TBL_CT05);
                    float cl = bilinear_interp(a3, ct, CL_O_TBL_CT0, CL_O_TBL_CT05)
                               + cl_de * de
                               + bilinear_interp(a3, ct, CL_Q_TBL_CT0, CL_Q_TBL_CT05) * q_hat;
                    float cd = bilinear_interp(a3, ct, CD_O_TBL_CT0, CD_O_TBL_CT05);
                    float cm = bilinear_interp(a3, ct, CM_O_TBL_CT0, CM_O_TBL_CT05)
                               + cm_de * de
                               + bilinear_interp(a3, ct, CM_Q_TBL_CT0, CM_Q_TBL_CT05) * q_hat;
                    float qS = 0.5f * RHO * S * vt * vt;
                    float L = qS * cl;
                    float D = qS * cd;
                    float My = qS * CHORD * cm;
                    k3_g = L / (MASS * vt) - (G / vt) * cosf(g3);
                    k3_v = (-G * sinf(g3) - D / MASS) / v_stall;
                    k3_a = q3 - k3_g;
                    k3_q = My / I_YY;
                }
                // --- k4 ---
                {
                    float g4 = gamma + dt_micro*k3_g, v4 = vn + dt_micro*k3_v;
                    float a4 = alpha + dt_micro*k3_a, q4 = q + dt_micro*k3_q;
                    float vt = fmaxf(v4 * v_stall, 0.1f);
                    float q_hat = q4 * CHORD / (2.0f * vt);
                    float ct = compute_ct(throttle, vt, k_thrust);
                    float cl_de = bilinear_interp(a4, ct, CL_DE_TBL_CT0, CL_DE_TBL_CT05);
                    float cm_de = bilinear_interp(a4, ct, CM_DE_TBL_CT0, CM_DE_TBL_CT05);
                    float cl = bilinear_interp(a4, ct, CL_O_TBL_CT0, CL_O_TBL_CT05)
                               + cl_de * de
                               + bilinear_interp(a4, ct, CL_Q_TBL_CT0, CL_Q_TBL_CT05) * q_hat;
                    float cd = bilinear_interp(a4, ct, CD_O_TBL_CT0, CD_O_TBL_CT05);
                    float cm = bilinear_interp(a4, ct, CM_O_TBL_CT0, CM_O_TBL_CT05)
                               + cm_de * de
                               + bilinear_interp(a4, ct, CM_Q_TBL_CT0, CM_Q_TBL_CT05) * q_hat;
                    float qS = 0.5f * RHO * S * vt * vt;
                    float L = qS * cl;
                    float D = qS * cd;
                    float My = qS * CHORD * cm;
                    k4_g = L / (MASS * vt) - (G / vt) * cosf(g4);
                    k4_v = (-G * sinf(g4) - D / MASS) / v_stall;
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
                // Thresholds aligned with Riley (1985): 14 deg = 0.24435 rad, -10 deg = -0.17453 rad
                if (alpha > 0.24435f) {
                    total_reward -= W_ALPHA_BARRIER_POS * (alpha - 0.24435f) * dt_micro;
                } else if (alpha < -0.17453f) {
                    total_reward -= W_ALPHA_BARRIER_NEG * (-alpha - 0.17453f) * dt_micro;
                }

                if (alpha >= 0.698132f || alpha <= -0.698132f || gamma <= -3.09159f) {
                    total_reward -= W_CRASH_PENALTY * v_stall; 
                    break;
                }
            }
            
            // 4. Control Effort Penalty
            total_reward -= W_CONTROL_EFFORT * (de * de) * (dt_micro * n_micro);
            total_reward += W_THROTTLE_BONUS * throttle * fmaxf(1.0f - vn, 0.0f) * (dt_micro * n_micro);
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
