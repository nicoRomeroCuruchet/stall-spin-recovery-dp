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


# Fused reduction kernel: max(abs(A - B)) with zero auxiliary VRAM
max_abs_diff_kernel = cp.ReductionKernel(
    in_params='float32 x, float32 y',
    out_params='float32 z',
    map_expr='abs(x - y)',
    reduce_expr='max(a, b)',
    post_map_expr='z = a',
    identity='0.0f',
    name='max_abs_diff_banked_spin'
)


@dataclass
class PolicyIterationBankedSpinConfig:
    """
    Configuration parameters for the 6-DOF Banked-Spin Policy Iteration (A.i).

    State space: (γ, V/Vs, α, μ, p, q)
    Action space: (δe, δa, δt)
    """
    maximum_iterations: int = 20_000
    gamma: float = 1.0
    theta: float = 1e-4
    n_steps: int = 100
    log: bool = False
    log_interval: int = 150
    img_path: Path = field(default_factory=lambda: Path("./img"))

    # --- Reward Shaping Weights (CUDA #defines) ---
    # Longitudinal (mirror PolicyIterationStallConfig)
    w_q_penalty: float = 2.0
    w_alpha_barrier_pos: float = 100.0
    w_alpha_barrier_neg: float = 10.0
    w_crash_penalty: float = 1000.0
    w_control_effort: float = 1.0
    w_throttle_bonus: float = 0.2

    # Lateral (new — must match aircraft/banked_spin.py shaping)
    w_p_penalty: float = 0.01
    w_mu_barrier: float = 0.5
    w_aileron_effort: float = 0.001


class PolicyIterationBankedSpin:
    """
    High-performance Procedural Policy Iteration for 6-DOF Banked-Spin Recovery (A.i).

    State space: (γ, V/Vs, α, μ, p, q)
    Action space: (δe, δa, δt)

    Embeds the full Grumman AA-1 Yankee aerodynamic model (longitudinal CL/CD/Cm
    and lateral Cl rolling-moment from Riley Table III(f)) and 6D barycentric
    interpolation directly into CUDA C++ kernels. Sideslip and yaw rate are
    pinned to zero (A.i): side-force and yawing-moment tables are not used.
    """

    def __init__(
        self,
        env: gym.Env,
        states_space: np.ndarray,
        action_space: np.ndarray,
        config: PolicyIterationBankedSpinConfig,
    ) -> None:

        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for procedural on-the-fly CUDA kernels.")

        self.env = env
        self.states_space = np.ascontiguousarray(states_space, dtype=np.float32)
        self.action_space = np.ascontiguousarray(action_space, dtype=np.float32)
        self.config = config

        self.n_states, self.n_dims = self.states_space.shape
        if self.n_dims != 6:
            raise ValueError(f"Expected 6D state space, got {self.n_dims}D")
        self.n_actions = len(self.action_space)
        if self.action_space.shape[1] != 3:
            raise ValueError(f"Expected 3D action space, got {self.action_space.shape[1]}D")
        self.n_corners = 2 ** self.n_dims  # 64 for 6D

        airplane = env.airplane
        self.v_stall = airplane.STALL_AIRSPEED
        self.k_thrust = airplane.THROTTLE_LINEAR_MAPPING
        self.dt = airplane.TIME_STEP

        self._precompute_grid_metadata()
        self._allocate_tensors_and_compile()

    def _precompute_grid_metadata(self) -> None:
        """Extract bounds, shape, and strides for the 6D CUDA interpolation."""
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
        logger.info(
            f"6D grid precomputed. Shape: {self.grid_shape}, Strides: {self.strides}, "
            f"States: {self.n_states:,}, Actions: {self.n_actions:,}"
        )

    def _allocate_tensors_and_compile(self) -> None:
        """Allocate VRAM tensors and compile the 6-DOF CUDA kernels."""
        logger.info("Allocating procedural tensors and compiling 6-DOF CUDA JIT Kernels...")

        self.d_states = cp.asarray(self.states_space, dtype=cp.float32)
        self.d_actions = cp.asarray(self.action_space, dtype=cp.float32)
        self.d_bounds_low = cp.asarray(self.bounds_low, dtype=cp.float32)
        self.d_bounds_high = cp.asarray(self.bounds_high, dtype=cp.float32)
        self.d_grid_shape = cp.asarray(self.grid_shape, dtype=cp.int32)
        self.d_strides = cp.asarray(self.strides, dtype=cp.int32)

        self.d_policy = cp.zeros(self.n_states, dtype=cp.int32)
        self.d_value_function = cp.zeros(self.n_states, dtype=cp.float32)
        self.d_new_value_function = cp.zeros(self.n_states, dtype=cp.float32)

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
        logger.success("6-DOF CUDA Kernels compiled. VRAM usage optimized.")

    def _compile_cuda_module(self) -> None:
        """Compile the 6-DOF aerodynamics + Bellman operations into CUDA."""
        cfg = self.config
        reward_defines = f'''
        #define W_Q_PENALTY {cfg.w_q_penalty:.6f}f
        #define W_ALPHA_BARRIER_POS {cfg.w_alpha_barrier_pos:.6f}f
        #define W_ALPHA_BARRIER_NEG {cfg.w_alpha_barrier_neg:.6f}f
        #define W_CRASH_PENALTY {cfg.w_crash_penalty:.6f}f
        #define W_CONTROL_EFFORT {cfg.w_control_effort:.6f}f
        #define W_THROTTLE_BONUS {cfg.w_throttle_bonus:.6f}f
        #define W_P_PENALTY {cfg.w_p_penalty:.6f}f
        #define W_MU_BARRIER {cfg.w_mu_barrier:.6f}f
        #define W_AILERON_EFFORT {cfg.w_aileron_effort:.6f}f
        #define MU_BARRIER 1.047197f          /* 60 deg in rad */
        #define MU_CRASH   1.5707963f         /* 90 deg in rad */
        #define P_CRASH    3.0f
        #define ALPHA_HI   0.698132f          /* +40 deg */
        #define ALPHA_LO  -0.698132f          /* -40 deg */
        #define GAMMA_LO  -3.09159f           /* -pi + 0.05 */
        '''
        cuda_source = reward_defines + r'''
        extern "C" {

        __device__ const float MASS = 715.21f;
        __device__ const float S = 9.1147f;
        __device__ const float CHORD = 1.22f;
        __device__ const float SPAN = 8.066f;     // Riley Table I: 26.46 ft x 0.3048
        __device__ const float RHO = 1.225f;
        __device__ const float G = 9.81f;
        __device__ const float I_XX = 808.06f;    // Riley Table I
        __device__ const float I_YY = 1000.60f;   // Riley Table I: 738 slug-ft^2 x 1.35582

        // Riley (1985) Table III - alpha breakpoints (radians)
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
            float vt_clip = fmaxf(vt, 0.1f);
            float q_bar = 0.5f * RHO * vt_clip * vt_clip;
            return fminf(fmaxf(k_thrust * throttle / (q_bar * S), 0.0f), 0.5f);
        }

        // ============================================================
        // Riley (1985) Table III(a-c) - Longitudinal aero
        // ============================================================

        __device__ const float CL_O_TBL_CT0[14] = {
            -0.41f, -0.01f,  0.41f,  0.84f,  1.16f,  1.23f,  1.26f,
             1.26f,  1.26f,  1.25f,  1.22f,  1.17f,  1.13f,  1.08f
        };
        __device__ const float CL_O_TBL_CT05[14] = {
            -0.67f, -0.14f,  0.41f,  0.97f,  1.42f,  1.54f,  1.62f,
             1.67f,  1.72f,  1.76f,  1.85f,  1.92f,  1.99f,  2.05f
        };
        __device__ const float CL_Q_TBL_CT0[14] = {
             2.41f,  2.41f,  2.42f,  2.46f,  2.59f,  2.96f,  3.72f,
             4.73f,  5.29f,  5.16f,  5.05f,  5.06f,  5.08f,  5.08f
        };
        __device__ const float CL_Q_TBL_CT05[14] = {
             3.012f, 3.012f, 3.029f, 3.222f, 3.594f, 4.351f, 6.072f,
             6.382f, 6.988f, 6.833f, 6.561f, 6.127f, 5.966f, 5.811f
        };
        __device__ const float CD_O_TBL_CT0[14] = {
            0.0666f, 0.0486f, 0.0526f, 0.0846f, 0.1456f, 0.1856f, 0.2446f,
            0.3136f, 0.3786f, 0.4486f, 0.6186f, 0.7786f, 0.9255f, 1.0636f
        };
        __device__ const float CD_O_TBL_CT05[14] = {
            -0.3273f, -0.3499f, -0.3474f, -0.3139f, -0.2483f, -0.2057f, -0.1435f,
            -0.0709f, -0.0018f,  0.0727f,  0.2561f,  0.4322f,  0.5979f,  0.7572f
        };
        __device__ const float CM_O_TBL_CT0[14] = {
             0.2700f,  0.1580f,  0.0760f,  0.0020f, -0.0800f, -0.1180f, -0.1670f,
            -0.2250f, -0.2770f, -0.3160f, -0.4080f, -0.4800f, -0.5560f, -0.6060f
        };
        __device__ const float CM_O_TBL_CT05[14] = {
             0.2700f,  0.1580f,  0.0760f,  0.0020f, -0.0800f, -0.1180f, -0.1670f,
            -0.2250f, -0.2770f, -0.3160f, -0.4080f, -0.4800f, -0.5560f, -0.6060f
        };
        __device__ const float CM_Q_TBL_CT0[14] = {
             -7.0000f,  -7.0000f,  -7.0400f,  -7.1500f,  -7.5200f,  -8.6200f, -10.8000f,
            -13.7300f, -15.3800f, -15.0000f, -14.6600f, -14.7100f, -14.7700f, -14.7700f
        };
        __device__ const float CM_Q_TBL_CT05[14] = {
             -8.75f,  -8.75f,  -8.80f,  -9.36f, -10.44f, -12.64f, -17.64f,
            -18.54f, -20.30f, -19.85f, -19.06f, -17.80f, -17.33f, -16.88f
        };
        __device__ const float CL_DE_TBL_CT0[14] = {
            0.35523f, 0.36096f, 0.35523f, 0.33232f, 0.30367f, 0.29221f, 0.28648f,
            0.28075f, 0.27502f, 0.26929f, 0.25210f, 0.24064f, 0.22345f, 0.21192f
        };
        __device__ const float CL_DE_TBL_CT05[14] = {
            0.79641f, 0.76776f, 0.75057f, 0.70474f, 0.62453f, 0.59588f, 0.57869f,
            0.56150f, 0.53858f, 0.51566f, 0.45837f, 0.41826f, 0.34950f, 0.31513f
        };
        __device__ const float CM_DE_TBL_CT0[14] = {
            -1.10581f, -1.10581f, -1.10581f, -1.03132f, -0.94538f, -0.93965f, -0.93392f,
            -0.92819f, -0.92819f, -0.92819f, -0.92819f, -0.85944f, -0.74485f, -0.57296f
        };
        __device__ const float CM_DE_TBL_CT05[14] = {
            -2.14484f, -2.25173f, -2.25746f, -2.26319f, -2.20016f, -2.06265f, -1.91348f,
            -1.78190f, -1.65032f, -1.54146f, -1.29488f, -1.22040f, -1.08862f, -0.87662f
        };

        // ============================================================
        // Riley (1985) Table III(f) - Rolling-moment derivatives
        // (only Cl_o, Cl_p_hat and Cl_da are needed under A.i: beta=r=dr=0)
        // ============================================================

        __device__ const float CL_ROLL_O_TBL_CT0[14] = {
             0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f, -0.0025f,
            -0.0050f, -0.0075f, -0.0075f, -0.0075f, -0.0075f, -0.0075f, -0.0075f
        };
        __device__ const float CL_ROLL_O_TBL_CT05[14] = {
             0.0060f,  0.0040f,  0.0020f,  0.0000f,  0.0000f,  0.0000f, -0.0025f,
            -0.0050f, -0.0075f, -0.0075f, -0.0075f, -0.0095f, -0.0115f, -0.0135f
        };
        __device__ const float CL_ROLL_PHAT_TBL_CT0[14] = {
            -0.5200f, -0.5200f, -0.5200f, -0.5200f, -0.4000f, -0.3100f, -0.2200f,
            -0.1300f, -0.0400f,  0.0500f,  0.0000f, -0.0500f, -0.1000f, -0.1500f
        };
        __device__ const float CL_ROLL_PHAT_TBL_CT05[14] = {
            -0.5200f, -0.5200f, -0.5200f, -0.5200f, -0.4000f, -0.3100f, -0.2200f,
            -0.1300f, -0.0400f,  0.0500f,  0.0000f, -0.0500f, -0.1000f, -0.1500f
        };
        // CT-independent (per Riley table layout); already in /rad
        __device__ const float CL_ROLL_DA_TBL[14] = {
            -0.05959f, -0.05959f, -0.05959f, -0.05730f, -0.05271f, -0.05042f, -0.04813f,
            -0.04527f, -0.04240f, -0.03953f, -0.03438f, -0.02865f, -0.02292f, -0.01891f
        };

        // ============================================================
        // Derivative computation (single state -> 6 derivatives)
        // ============================================================

        __device__ void compute_derivs_6dof(
            float gamma, float vn, float alpha, float mu, float p, float q,
            float de, float da, float throttle, float v_stall, float k_thrust,
            float* g_dot, float* v_dot, float* a_dot,
            float* m_dot, float* p_dot, float* q_dot
        ) {
            float vt = fmaxf(vn * v_stall, 0.1f);
            float q_hat = q * CHORD / (2.0f * vt);
            float p_hat = p * SPAN  / (2.0f * vt);
            float ct = compute_ct(throttle, vt, k_thrust);

            // Longitudinal aero (Riley III(a-c))
            float cl_de = bilinear_interp(alpha, ct, CL_DE_TBL_CT0, CL_DE_TBL_CT05);
            float cm_de = bilinear_interp(alpha, ct, CM_DE_TBL_CT0, CM_DE_TBL_CT05);
            float cl = bilinear_interp(alpha, ct, CL_O_TBL_CT0, CL_O_TBL_CT05)
                     + cl_de * de
                     + bilinear_interp(alpha, ct, CL_Q_TBL_CT0, CL_Q_TBL_CT05) * q_hat;
            float cd = bilinear_interp(alpha, ct, CD_O_TBL_CT0, CD_O_TBL_CT05);
            float cm = bilinear_interp(alpha, ct, CM_O_TBL_CT0, CM_O_TBL_CT05)
                     + cm_de * de
                     + bilinear_interp(alpha, ct, CM_Q_TBL_CT0, CM_Q_TBL_CT05) * q_hat;

            // Lateral rolling moment (Riley III(f) - A.i)
            float cl_roll_o = bilinear_interp(alpha, ct, CL_ROLL_O_TBL_CT0, CL_ROLL_O_TBL_CT05);
            float cl_roll_p = bilinear_interp(alpha, ct, CL_ROLL_PHAT_TBL_CT0, CL_ROLL_PHAT_TBL_CT05);
            float cl_roll_da = cl_interp(alpha, CL_ROLL_DA_TBL);
            float cl_roll = cl_roll_o + cl_roll_p * p_hat + cl_roll_da * da;

            float qS  = 0.5f * RHO * S * vt * vt;
            float L   = qS * cl;
            float D   = qS * cd;
            float My  = qS * CHORD * cm;
            float Lr  = qS * SPAN  * cl_roll;

            float cos_g = cosf(gamma);
            float sin_g = sinf(gamma);
            float cos_m = cosf(mu);
            float sin_m = sinf(mu);
            float tan_g = tanf(gamma);
            float cos_a = cosf(alpha);

            *g_dot = (L * cos_m) / (MASS * vt) - (G / vt) * cos_g;
            *v_dot = (-G * sin_g - D / MASS) / v_stall;
            *a_dot = q - L / (MASS * vt) + (G / vt) * cos_g * cos_m;
            *m_dot = p * cos_a + sin_m * tan_g * L / (MASS * vt);
            *p_dot = Lr / I_XX;
            *q_dot = My / I_YY;
        }

        // ============================================================
        // RK4 step over 6 states + Markov-compliant rewards
        // ============================================================

        __device__ void rk4_step_6dof(
            float& gamma, float& vn, float& alpha, float& mu, float& p, float& q,
            float de, float da, float throttle, float dt_micro, int n_micro,
            float v_stall, float k_thrust, float& total_reward
        ) {
            total_reward = 0.0f;

            for (int m = 0; m < n_micro; ++m) {
                float k1g, k1v, k1a, k1m, k1p, k1q;
                float k2g, k2v, k2a, k2m, k2p, k2q;
                float k3g, k3v, k3a, k3m, k3p, k3q;
                float k4g, k4v, k4a, k4m, k4p, k4q;

                compute_derivs_6dof(
                    gamma, vn, alpha, mu, p, q,
                    de, da, throttle, v_stall, k_thrust,
                    &k1g, &k1v, &k1a, &k1m, &k1p, &k1q);

                compute_derivs_6dof(
                    gamma + 0.5f*dt_micro*k1g, vn + 0.5f*dt_micro*k1v,
                    alpha + 0.5f*dt_micro*k1a, mu + 0.5f*dt_micro*k1m,
                    p + 0.5f*dt_micro*k1p, q + 0.5f*dt_micro*k1q,
                    de, da, throttle, v_stall, k_thrust,
                    &k2g, &k2v, &k2a, &k2m, &k2p, &k2q);

                compute_derivs_6dof(
                    gamma + 0.5f*dt_micro*k2g, vn + 0.5f*dt_micro*k2v,
                    alpha + 0.5f*dt_micro*k2a, mu + 0.5f*dt_micro*k2m,
                    p + 0.5f*dt_micro*k2p, q + 0.5f*dt_micro*k2q,
                    de, da, throttle, v_stall, k_thrust,
                    &k3g, &k3v, &k3a, &k3m, &k3p, &k3q);

                compute_derivs_6dof(
                    gamma + dt_micro*k3g, vn + dt_micro*k3v,
                    alpha + dt_micro*k3a, mu + dt_micro*k3m,
                    p + dt_micro*k3p, q + dt_micro*k3q,
                    de, da, throttle, v_stall, k_thrust,
                    &k4g, &k4v, &k4a, &k4m, &k4p, &k4q);

                gamma += (dt_micro / 6.0f) * (k1g + 2.0f*k2g + 2.0f*k3g + k4g);
                vn    += (dt_micro / 6.0f) * (k1v + 2.0f*k2v + 2.0f*k3v + k4v);
                alpha += (dt_micro / 6.0f) * (k1a + 2.0f*k2a + 2.0f*k3a + k4a);
                mu    += (dt_micro / 6.0f) * (k1m + 2.0f*k2m + 2.0f*k3m + k4m);
                p     += (dt_micro / 6.0f) * (k1p + 2.0f*k2p + 2.0f*k3p + k4p);
                q     += (dt_micro / 6.0f) * (k1q + 2.0f*k2q + 2.0f*k3q + k4q);

                // === Markov-compliant rewards ===
                // 1. Primary cost: altitude loss (signed)
                total_reward += dt_micro * vn * v_stall * sinf(gamma);
                if (gamma >= 0.0f) { break; }

                // 2. Pitch damping
                total_reward -= W_Q_PENALTY * (q * q) * dt_micro;

                // 3. Markovian alpha barrier (Riley stall thresholds)
                if (alpha > 0.24435f) {
                    total_reward -= W_ALPHA_BARRIER_POS * (alpha - 0.24435f) * dt_micro;
                } else if (alpha < -0.17453f) {
                    total_reward -= W_ALPHA_BARRIER_NEG * (-alpha - 0.17453f) * dt_micro;
                }

                // 4. Roll-rate damping
                total_reward -= W_P_PENALTY * (p * p) * dt_micro;

                // 5. Bank-angle soft barrier (kicks in past 60 deg)
                float abs_mu = fabsf(mu);
                if (abs_mu > MU_BARRIER) {
                    float excess = abs_mu - MU_BARRIER;
                    total_reward -= W_MU_BARRIER * excess * excess * dt_micro;
                }

                // 6. Crash conditions (longitudinal + lateral)
                if (alpha >= ALPHA_HI || alpha <= ALPHA_LO || gamma <= GAMMA_LO
                    || abs_mu >= MU_CRASH || fabsf(p) >= P_CRASH) {
                    total_reward -= W_CRASH_PENALTY * v_stall;
                    break;
                }
            }

            // Per-action effort terms (single application per macro-step)
            total_reward -= W_CONTROL_EFFORT * (de * de) * (dt_micro * n_micro);
            total_reward -= W_AILERON_EFFORT * fabsf(da) * (dt_micro * n_micro);
            total_reward += W_THROTTLE_BONUS * throttle * fmaxf(1.0f - vn, 0.0f) * (dt_micro * n_micro);
        }

        // ============================================================
        // 6D Barycentric interpolation (64 corners)
        // ============================================================

        __device__ void get_barycentric_6d(
            float s0, float s1, float s2, float s3, float s4, float s5,
            const float* b_low, const float* b_high, const int* g_shape, const int* strides,
            int* idxs, float* wgts
        ) {
            float n[6];
            int   i[6];
            float d[6];

            n[0] = (s0 - b_low[0]) / (b_high[0] - b_low[0]) * (g_shape[0] - 1);
            n[1] = (s1 - b_low[1]) / (b_high[1] - b_low[1]) * (g_shape[1] - 1);
            n[2] = (s2 - b_low[2]) / (b_high[2] - b_low[2]) * (g_shape[2] - 1);
            n[3] = (s3 - b_low[3]) / (b_high[3] - b_low[3]) * (g_shape[3] - 1);
            n[4] = (s4 - b_low[4]) / (b_high[4] - b_low[4]) * (g_shape[4] - 1);
            n[5] = (s5 - b_low[5]) / (b_high[5] - b_low[5]) * (g_shape[5] - 1);

            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                n[k] = fmaxf(0.0f, fminf(n[k], (float)(g_shape[k] - 1)));
                i[k] = (int)n[k];
                if (i[k] == g_shape[k] - 1) i[k]--;
                d[k] = n[k] - i[k];
            }

            #pragma unroll
            for (int corner = 0; corner < 64; ++corner) {
                int b0 = (corner >> 5) & 1;
                int b1 = (corner >> 4) & 1;
                int b2 = (corner >> 3) & 1;
                int b3 = (corner >> 2) & 1;
                int b4 = (corner >> 1) & 1;
                int b5 =  corner       & 1;

                idxs[corner] = (i[0] + b0) * strides[0] + (i[1] + b1) * strides[1]
                             + (i[2] + b2) * strides[2] + (i[3] + b3) * strides[3]
                             + (i[4] + b4) * strides[4] + (i[5] + b5) * strides[5];

                wgts[corner] = (b0 ? d[0] : (1.0f - d[0]))
                             * (b1 ? d[1] : (1.0f - d[1]))
                             * (b2 ? d[2] : (1.0f - d[2]))
                             * (b3 ? d[3] : (1.0f - d[3]))
                             * (b4 ? d[4] : (1.0f - d[4]))
                             * (b5 ? d[5] : (1.0f - d[5]));
            }
        }

        // ============================================================
        // Bellman kernels
        // ============================================================

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
            float gamma = states[s_idx * 6 + 0];
            float vn    = states[s_idx * 6 + 1];
            float alpha = states[s_idx * 6 + 2];
            float mu    = states[s_idx * 6 + 3];
            float p     = states[s_idx * 6 + 4];
            float q     = states[s_idx * 6 + 5];
            float de       = actions[a_idx * 3 + 0];
            float da       = actions[a_idx * 3 + 1];
            float throttle = actions[a_idx * 3 + 2];
            float reward;

            rk4_step_6dof(gamma, vn, alpha, mu, p, q,
                          de, da, throttle, dt, 1, v_stall, k_thrust, reward);

            int idxs[64]; float wgts[64];
            get_barycentric_6d(gamma, vn, alpha, mu, p, q,
                               b_low, b_high, g_shape, strides, idxs, wgts);

            float expected_v = 0.0f;
            #pragma unroll
            for (int i = 0; i < 64; ++i) {
                expected_v = fmaf(wgts[i], V[idxs[i]], expected_v);
            }
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

            float ig = states[s_idx * 6 + 0];
            float iv = states[s_idx * 6 + 1];
            float ia = states[s_idx * 6 + 2];
            float im = states[s_idx * 6 + 3];
            float ip = states[s_idx * 6 + 4];
            float iq = states[s_idx * 6 + 5];

            float max_q_val = -1e9f;
            int best_a = 0;

            for (int a = 0; a < n_actions; ++a) {
                float gamma = ig, vn = iv, alpha = ia, mu = im, p = ip, q = iq;
                float de       = actions[a * 3 + 0];
                float da       = actions[a * 3 + 1];
                float throttle = actions[a * 3 + 2];
                float reward;

                rk4_step_6dof(gamma, vn, alpha, mu, p, q,
                              de, da, throttle, dt, 1, v_stall, k_thrust, reward);

                int idxs[64]; float wgts[64];
                get_barycentric_6d(gamma, vn, alpha, mu, p, q,
                                   b_low, b_high, g_shape, strides, idxs, wgts);

                float expected_v = 0.0f;
                #pragma unroll
                for (int i = 0; i < 64; ++i) {
                    expected_v = fmaf(wgts[i], V[idxs[i]], expected_v);
                }

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
        # Lower thread count vs 4-DOF: idxs[64]+wgts[64] increase register/local-mem pressure
        self.threads_per_block = 128
        self.blocks_per_grid = (
            (self.n_states + self.threads_per_block - 1) // self.threads_per_block
        )

    def _pull_tensors_from_gpu(self) -> None:
        """Retrieve converged value/policy to host RAM."""
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
            self.d_value_function, self.d_new_value_function = (
                self.d_new_value_function, self.d_value_function
            )

            if i % SYNC_INTERVAL == 0 or i == self.config.maximum_iterations - 1:
                delta = float(d_delta.get())
                if delta < self.config.theta:
                    logger.success(f"GPU Evaluation converged at step {i} with Δ={delta:.5e}")
                    return delta

        logger.warning(
            f"GPU Evaluation hit max iterations ({self.config.maximum_iterations}) "
            f"with Δ={delta:.5e}"
        )
        return delta

    def policy_improvement(self) -> bool:
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
        # Allow ~0.01% chatter (float-precision oscillation between adjacent actions)
        tolerance_threshold = int(self.n_states * 0.0001)
        policy_stable = (changes <= tolerance_threshold)

        if not policy_stable:
            logger.info(
                f"GPU Policy updated: {changes} states changed optimal action. "
                f"(Tolerance: {tolerance_threshold})"
            )
        return policy_stable

    def run(self) -> None:
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
            action_space=self.action_space,
        )
        logger.success(f"Policy saved successfully to {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: Path, env: gym.Env = None) -> "PolicyIterationBankedSpin":
        filepath = filepath.with_suffix(".npz")
        logger.info(f"Loading policy from {filepath.resolve()}...")
        data = np.load(filepath)

        instance = cls.__new__(cls)
        instance.env = env
        instance.config = PolicyIterationBankedSpinConfig()

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
