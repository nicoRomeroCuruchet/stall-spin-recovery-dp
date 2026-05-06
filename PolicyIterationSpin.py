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


max_abs_diff_kernel = cp.ReductionKernel(
    in_params='float32 x, float32 y',
    out_params='float32 z',
    map_expr='abs(x - y)',
    reduce_expr='max(a, b)',
    post_map_expr='z = a',
    identity='0.0f',
    name='max_abs_diff_spin',
)


@dataclass
class PolicyIterationSpinConfig:
    """
    Configuration for the 8-DOF Spin Recovery Policy Iteration.

    State: (γ, V/Vs, α, β, μ, p, q, r)
    Action: (δe, δa, δt, δr)
    """
    maximum_iterations: int = 8000
    gamma: float = 1.0
    theta: float = 5e-4
    n_steps: int = 50
    log: bool = False
    log_interval: int = 10
    img_path: Path = field(default_factory=lambda: Path("./img"))

    # Reward shaping (CUDA #defines, must match aircraft/spin.py)
    w_q_penalty: float = 2.0
    w_alpha_barrier_pos: float = 100.0
    w_alpha_barrier_neg: float = 10.0
    w_crash_penalty: float = 1000.0
    w_control_effort: float = 10.0
    w_throttle_bonus: float = 0.2

    w_p_penalty: float = 0.1
    w_r_penalty: float = 0.1
    w_beta_penalty: float = 0.5
    w_mu_barrier: float = 0.5
    w_aileron_effort: float = 5.0
    w_rudder_effort: float = 5.0


class PolicyIterationSpin:
    """
    High-performance Procedural Policy Iteration for 8-DOF Spin Recovery.

    Embeds Riley (1985) III(a-f) full aero tables and the stability-axis
    8-DOF dynamics (with `I_xz = 0` and propeller gyroscopic) directly
    into CUDA kernels. 8D barycentric uses 256 corners per state.
    """

    def __init__(
        self,
        env: gym.Env,
        states_space: np.ndarray,
        action_space: np.ndarray,
        config: PolicyIterationSpinConfig,
    ) -> None:

        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for procedural CUDA kernels.")

        self.env = env
        self.states_space = np.ascontiguousarray(states_space, dtype=np.float32)
        self.action_space = np.ascontiguousarray(action_space, dtype=np.float32)
        self.config = config

        self.n_states, self.n_dims = self.states_space.shape
        if self.n_dims != 8:
            raise ValueError(f"Expected 8D state space, got {self.n_dims}D")
        self.n_actions = len(self.action_space)
        if self.action_space.shape[1] != 4:
            raise ValueError(f"Expected 4D action, got {self.action_space.shape[1]}D")
        self.n_corners = 2 ** self.n_dims  # 256 for 8D

        airplane = env.airplane
        self.v_stall = airplane.STALL_AIRSPEED
        self.k_thrust = airplane.THROTTLE_LINEAR_MAPPING
        self.dt = airplane.TIME_STEP

        self._precompute_grid_metadata()
        self._allocate_tensors_and_compile()

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

        self.corner_bits = np.array(
            list(product([0, 1], repeat=self.n_dims)), dtype=np.int32
        )
        logger.info(
            f"8D grid: shape={self.grid_shape}, strides={self.strides}, "
            f"states={self.n_states:,}, actions={self.n_actions:,}"
        )

    def _allocate_tensors_and_compile(self) -> None:
        logger.info("Allocating procedural tensors and compiling 8-DOF CUDA Kernels...")

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
        logger.success("8-DOF CUDA Kernels compiled.")

    def _compile_cuda_module(self) -> None:
        cfg = self.config
        reward_defines = f'''
        #define W_Q_PENALTY {cfg.w_q_penalty:.6f}f
        #define W_ALPHA_BARRIER_POS {cfg.w_alpha_barrier_pos:.6f}f
        #define W_ALPHA_BARRIER_NEG {cfg.w_alpha_barrier_neg:.6f}f
        #define W_CRASH_PENALTY {cfg.w_crash_penalty:.6f}f
        #define W_CONTROL_EFFORT {cfg.w_control_effort:.6f}f
        #define W_THROTTLE_BONUS {cfg.w_throttle_bonus:.6f}f
        #define W_P_PENALTY {cfg.w_p_penalty:.6f}f
        #define W_R_PENALTY {cfg.w_r_penalty:.6f}f
        #define W_BETA_PENALTY {cfg.w_beta_penalty:.6f}f
        #define W_MU_BARRIER {cfg.w_mu_barrier:.6f}f
        #define W_AILERON_EFFORT {cfg.w_aileron_effort:.6f}f
        #define W_RUDDER_EFFORT {cfg.w_rudder_effort:.6f}f
        #define MU_BARRIER 1.047197f
        #define MU_CRASH   1.5707963f
        #define P_CRASH    3.0f
        #define R_CRASH    4.0f
        #define BETA_CRASH 0.523599f
        #define ALPHA_HI   0.698132f
        #define ALPHA_LO  -0.698132f
        #define GAMMA_LO  -3.09159f
        '''
        cuda_source = reward_defines + r'''
        extern "C" {

        __device__ const float MASS = 715.21f;
        __device__ const float S = 9.1147f;
        __device__ const float CHORD = 1.22f;
        __device__ const float SPAN = 8.066f;
        __device__ const float RHO = 1.225f;
        __device__ const float G = 9.81f;
        __device__ const float I_XX = 808.06f;
        __device__ const float I_YY = 1000.60f;
        __device__ const float I_ZZ = 1719.18f;
        __device__ const float IP = 1.559f;
        __device__ const float N_MAX = 272.27f;

        __device__ const float ALPHA_TBL[14] = {
            -0.17453f, -0.08727f,  0.00000f,  0.08727f,  0.17453f,
             0.20944f,  0.24435f,  0.27925f,  0.31416f,  0.34907f,
             0.43633f,  0.52360f,  0.61087f,  0.69813f
        };

        __device__ float lin_interp(float alpha, const float* tbl) {
            if (alpha <= ALPHA_TBL[0])  return tbl[0];
            if (alpha >= ALPHA_TBL[13]) return tbl[13];
            for (int i = 0; i < 13; ++i) {
                if (alpha <= ALPHA_TBL[i+1]) {
                    float t = (alpha - ALPHA_TBL[i]) / (ALPHA_TBL[i+1] - ALPHA_TBL[i]);
                    return tbl[i] + t * (tbl[i+1] - tbl[i]);
                }
            }
            return tbl[13];
        }

        __device__ float bilin(float alpha, float ct,
                               const float* tbl0, const float* tbl05) {
            float t = fminf(fmaxf(ct / 0.5f, 0.0f), 1.0f);
            return lin_interp(alpha, tbl0)
                 + t * (lin_interp(alpha, tbl05) - lin_interp(alpha, tbl0));
        }

        __device__ float compute_ct(float throttle, float vt, float k_thrust) {
            float vt_clip = fmaxf(vt, 0.1f);
            float q_bar = 0.5f * RHO * vt_clip * vt_clip;
            return fminf(fmaxf(k_thrust * throttle / (q_bar * S), 0.0f), 0.5f);
        }

        // ====================================================================
        // Riley (1985) Table III(a-c) - Longitudinal aero
        // ====================================================================
        __device__ const float CL_O_CT0[14]   = { -0.41f,-0.01f,0.41f,0.84f,1.16f,1.23f,1.26f,1.26f,1.26f,1.25f,1.22f,1.17f,1.13f,1.08f };
        __device__ const float CL_O_CT05[14]  = { -0.67f,-0.14f,0.41f,0.97f,1.42f,1.54f,1.62f,1.67f,1.72f,1.76f,1.85f,1.92f,1.99f,2.05f };
        __device__ const float CL_Q_CT0[14]   = { 2.41f,2.41f,2.42f,2.46f,2.59f,2.96f,3.72f,4.73f,5.29f,5.16f,5.05f,5.06f,5.08f,5.08f };
        __device__ const float CL_Q_CT05[14]  = { 3.012f,3.012f,3.029f,3.222f,3.594f,4.351f,6.072f,6.382f,6.988f,6.833f,6.561f,6.127f,5.966f,5.811f };
        __device__ const float CD_O_CT0[14]   = { 0.0666f,0.0486f,0.0526f,0.0846f,0.1456f,0.1856f,0.2446f,0.3136f,0.3786f,0.4486f,0.6186f,0.7786f,0.9255f,1.0636f };
        __device__ const float CD_O_CT05[14]  = { -0.3273f,-0.3499f,-0.3474f,-0.3139f,-0.2483f,-0.2057f,-0.1435f,-0.0709f,-0.0018f,0.0727f,0.2561f,0.4322f,0.5979f,0.7572f };
        __device__ const float CM_O_CT0[14]   = { 0.2700f,0.1580f,0.0760f,0.0020f,-0.0800f,-0.1180f,-0.1670f,-0.2250f,-0.2770f,-0.3160f,-0.4080f,-0.4800f,-0.5560f,-0.6060f };
        __device__ const float CM_O_CT05[14]  = { 0.2700f,0.1580f,0.0760f,0.0020f,-0.0800f,-0.1180f,-0.1670f,-0.2250f,-0.2770f,-0.3160f,-0.4080f,-0.4800f,-0.5560f,-0.6060f };
        __device__ const float CM_Q_CT0[14]   = { -7.0000f,-7.0000f,-7.0400f,-7.1500f,-7.5200f,-8.6200f,-10.8000f,-13.7300f,-15.3800f,-15.0000f,-14.6600f,-14.7100f,-14.7700f,-14.7700f };
        __device__ const float CM_Q_CT05[14]  = { -8.75f,-8.75f,-8.80f,-9.36f,-10.44f,-12.64f,-17.64f,-18.54f,-20.30f,-19.85f,-19.06f,-17.80f,-17.33f,-16.88f };
        __device__ const float CL_DE_CT0[14]  = { 0.35523f,0.36096f,0.35523f,0.33232f,0.30367f,0.29221f,0.28648f,0.28075f,0.27502f,0.26929f,0.25210f,0.24064f,0.22345f,0.21192f };
        __device__ const float CL_DE_CT05[14] = { 0.79641f,0.76776f,0.75057f,0.70474f,0.62453f,0.59588f,0.57869f,0.56150f,0.53858f,0.51566f,0.45837f,0.41826f,0.34950f,0.31513f };
        __device__ const float CM_DE_CT0[14]  = { -1.10581f,-1.10581f,-1.10581f,-1.03132f,-0.94538f,-0.93965f,-0.93392f,-0.92819f,-0.92819f,-0.92819f,-0.92819f,-0.85944f,-0.74485f,-0.57296f };
        __device__ const float CM_DE_CT05[14] = { -2.14484f,-2.25173f,-2.25746f,-2.26319f,-2.20016f,-2.06265f,-1.91348f,-1.78190f,-1.65032f,-1.54146f,-1.29488f,-1.22040f,-1.08862f,-0.87662f };

        // ====================================================================
        // Riley (1985) Table III(f) - Rolling-moment (Cl)
        // ====================================================================
        __device__ const float CL_RO_CT0[14]   = { 0,0,0,0,0,0,-0.0025f,-0.0050f,-0.0075f,-0.0075f,-0.0075f,-0.0075f,-0.0075f,-0.0075f };
        __device__ const float CL_RO_CT05[14]  = { 0.0060f,0.0040f,0.0020f,0,0,0,-0.0025f,-0.0050f,-0.0075f,-0.0075f,-0.0075f,-0.0095f,-0.0115f,-0.0135f };
        __device__ const float CL_RB_CT0[14]   = { -0.0802f,-0.0659f,-0.0659f,-0.1089f,-0.1805f,-0.2092f,-0.2292f,-0.2406f,-0.2492f,-0.2578f,-0.2578f,-0.2406f,-0.2292f,-0.2235f };
        __device__ const float CL_RB_CT05[14]  = { -0.1564f,-0.1232f,-0.1043f,-0.1089f,-0.1369f,-0.1518f,-0.1530f,-0.1358f,-0.1215f,-0.1048f,-0.1243f,-0.2023f,-0.2676f,-0.2996f };
        __device__ const float CL_RP_CT0[14]   = { -0.5200f,-0.5200f,-0.5200f,-0.5200f,-0.4000f,-0.3100f,-0.2200f,-0.1300f,-0.0400f,0.0500f,0,-0.0500f,-0.1000f,-0.1500f };
        __device__ const float CL_RP_CT05[14]  = { -0.5200f,-0.5200f,-0.5200f,-0.5200f,-0.4000f,-0.3100f,-0.2200f,-0.1300f,-0.0400f,0.0500f,0,-0.0500f,-0.1000f,-0.1500f };
        __device__ const float CL_RR_CT0[14]   = { 0.1000f,0.1300f,0.1600f,0.1900f,0.1400f,0.1300f,0.1200f,0.1100f,0.1000f,0.0900f,0.0700f,0.0700f,0.0700f,0.1000f };
        __device__ const float CL_RR_CT05[14]  = { 0.1150f,0.1450f,0.1750f,0.2050f,0.1540f,0.1450f,0.1290f,0.1140f,0.1040f,0.0950f,0.0760f,0.0750f,0.0740f,0.1040f };
        __device__ const float CL_RDA[14]      = { -0.05959f,-0.05959f,-0.05959f,-0.05730f,-0.05271f,-0.05042f,-0.04813f,-0.04527f,-0.04240f,-0.03953f,-0.03438f,-0.02865f,-0.02292f,-0.01891f };
        __device__ const float CL_RDR[14]      = { 0.014324f,0.014324f,0.014324f,0.014324f,0.014324f,0.014324f,0.014324f,0.014324f,0.014324f,0.014324f,0.007448f,0,0,0 };

        // ====================================================================
        // Riley (1985) Table III(d) - Side-force (Cy)
        // ====================================================================
        __device__ const float CY_O_CT0[14]    = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
        __device__ const float CY_O_CT05[14]   = { 0.0810f,0.0540f,0.0270f,0,-0.0270f,-0.0378f,-0.0486f,-0.0540f,-0.0540f,-0.0540f,-0.0540f,-0.0540f,-0.0540f,-0.0540f };
        __device__ const float CY_B_CT0[14]    = { -0.7448f,-0.7162f,-0.6761f,-0.6303f,-0.6246f,-0.6189f,-0.5615f,-0.5042f,-0.4699f,-0.4469f,-0.3839f,-0.3438f,-0.3553f,-0.3896f };
        __device__ const float CY_B_CT05[14]   = { -1.2949f,-1.2949f,-1.2949f,-1.2949f,-1.2949f,-1.2949f,-1.2663f,-1.2204f,-1.2032f,-1.1917f,-1.1631f,-1.1574f,-1.2032f,-1.2720f };
        __device__ const float CY_DA[14]       = { -0.005730f,-0.004584f,-0.005157f,-0.005730f,-0.008021f,-0.008594f,-0.009167f,-0.007448f,-0.006303f,-0.005730f,-0.004584f,-0.005730f,0,0 };
        __device__ const float CY_DR[14]       = { -0.802142f,-0.229184f,0.343776f,0.916734f,1.489692f,1.718876f,1.948060f,2.177242f,2.406424f,2.635608f,3.208566f,3.781524f,1.890764f,0 };
        __device__ const float CY_P_CT0[14]    = { 0.00244f,0.00263f,0.00282f,0.00295f,0.00307f,0.00295f,0.00282f,0.00267f,0.00255f,0.00242f,0.00189f,0.00137f,0.00093f,0.00053f };
        __device__ const float CY_P_CT05[14]   = { 0.00589f,0.00629f,0.00674f,0.00722f,0.00773f,0.00775f,0.00777f,0.00777f,0.00779f,0.00779f,0.00665f,0.00558f,0.00425f,0.00295f };
        __device__ const float CY_R_CT0[14]    = { 0.8000f,0.9000f,1.0000f,1.1000f,0.8000f,0.6000f,0.4000f,0.2000f,0,-0.2500f,-0.2400f,-0.1200f,0,0 };
        __device__ const float CY_R_CT05[14]   = { 1.0110f,1.1110f,1.2110f,1.3110f,1.0010f,0.8020f,0.5290f,0.2490f,0.0560f,-0.1870f,-0.1680f,-0.0450f,0.0610f,0.0520f };

        // ====================================================================
        // Riley (1985) Table III(e) - Yawing-moment (Cn)
        // ====================================================================
        __device__ const float CN_O_CT0[14]    = { 0,0,0,0,0,0,-0.0010f,-0.0010f,-0.0010f,-0.0010f,-0.0010f,-0.0010f,-0.0010f,-0.0010f };
        __device__ const float CN_O_CT05[14]   = { -0.0166f,-0.0166f,-0.0166f,-0.0166f,-0.0166f,-0.0166f,-0.0142f,-0.0118f,-0.0094f,-0.0070f,-0.0010f,-0.0040f,-0.0070f,-0.0100f };
        __device__ const float CN_B_CT0[14]    = { 0.143240f,0.126051f,0.110008f,0.100269f,0.081360f,0.073339f,0.063026f,0.051566f,0.045837f,0.040107f,0.018335f,-0.001146f,-0.014324f,-0.021772f };
        __device__ const float CN_B_CT05[14]   = { 0.187358f,0.174180f,0.167305f,0.164440f,0.151835f,0.146678f,0.138657f,0.130063f,0.126625f,0.123761f,0.108864f,0.095686f,0.089383f,0.088237f };
        __device__ const float CN_DA[14]       = { 0.005157f,0.004011f,0.002865f,0.001719f,0.000573f,0,-0.001719f,-0.003438f,-0.005730f,-0.008594f,-0.005157f,-0.002292f,0,0.001719f };
        __device__ const float CN_DR_CT0[14]   = { -0.2000f,-0.2000f,-0.2000f,-0.2000f,-0.2000f,-0.2000f,-0.1300f,-0.0500f,-0.0600f,-0.0700f,-0.1000f,-0.1000f,-0.1000f,-0.1000f };
        __device__ const float CN_DR_CT05[14]  = { -0.2900f,-0.2900f,-0.2900f,-0.2900f,-0.2870f,-0.2860f,-0.1850f,-0.0710f,-0.0840f,-0.0970f,-0.1350f,-0.1320f,-0.1260f,-0.1220f };
        __device__ const float CN_P_CT0[14]    = { -0.0300f,-0.0400f,-0.0500f,-0.0600f,-0.0700f,-0.0600f,-0.0300f,0,0.0300f,0.0400f,0.0150f,0.0150f,0.0150f,0.0150f };
        __device__ const float CN_P_CT05[14]   = { -0.0480f,-0.0580f,-0.0680f,-0.0780f,-0.0880f,-0.0770f,-0.0470f,-0.0160f,0.0140f,0.0400f,0.0150f,0.0150f,0.0100f,0.0070f };
        __device__ const float CN_R_CT0[14]    = { -0.00116f,-0.00125f,-0.00134f,-0.00140f,-0.00146f,-0.00140f,-0.00134f,-0.00127f,-0.00121f,-0.00115f,-0.00090f,-0.00065f,-0.00044f,-0.00025f };
        __device__ const float CN_R_CT05[14]   = { -0.00280f,-0.00299f,-0.00320f,-0.00343f,-0.00367f,-0.00368f,-0.00369f,-0.00369f,-0.00370f,-0.00370f,-0.00316f,-0.00265f,-0.00202f,-0.00140f };

        // ====================================================================
        // 8-DOF derivative computation
        // ====================================================================
        __device__ void derivs8(
            float gamma, float vn, float alpha, float beta,
            float mu, float p, float q, float r,
            float de, float da, float throttle, float dr,
            float v_stall, float k_thrust,
            float* g_dot, float* v_dot, float* a_dot, float* b_dot,
            float* m_dot, float* p_dot, float* q_dot, float* r_dot
        ) {
            float vt = fmaxf(vn * v_stall, 0.1f);
            float q_hat = q * CHORD / (2.0f * vt);
            float p_hat = p * SPAN  / (2.0f * vt);
            float r_hat = r * SPAN  / (2.0f * vt);
            float ct = compute_ct(throttle, vt, k_thrust);
            float ip_n = IP * N_MAX * throttle;

            // Longitudinal coefficients
            float cl_de = bilin(alpha, ct, CL_DE_CT0, CL_DE_CT05);
            float cm_de = bilin(alpha, ct, CM_DE_CT0, CM_DE_CT05);
            float cl_lift = bilin(alpha, ct, CL_O_CT0, CL_O_CT05)
                          + cl_de * de
                          + bilin(alpha, ct, CL_Q_CT0, CL_Q_CT05) * q_hat;
            float cd = bilin(alpha, ct, CD_O_CT0, CD_O_CT05);
            float cm = bilin(alpha, ct, CM_O_CT0, CM_O_CT05)
                     + cm_de * de
                     + bilin(alpha, ct, CM_Q_CT0, CM_Q_CT05) * q_hat;

            // Lateral coefficients (full Riley III(d-f))
            float cy = bilin(alpha, ct, CY_O_CT0, CY_O_CT05)
                     + bilin(alpha, ct, CY_B_CT0, CY_B_CT05) * beta
                     + bilin(alpha, ct, CY_P_CT0, CY_P_CT05) * p_hat
                     + bilin(alpha, ct, CY_R_CT0, CY_R_CT05) * r_hat
                     + lin_interp(alpha, CY_DA) * da
                     + lin_interp(alpha, CY_DR) * dr;

            float cl_roll = bilin(alpha, ct, CL_RO_CT0, CL_RO_CT05)
                          + bilin(alpha, ct, CL_RB_CT0, CL_RB_CT05) * beta
                          + bilin(alpha, ct, CL_RP_CT0, CL_RP_CT05) * p_hat
                          + bilin(alpha, ct, CL_RR_CT0, CL_RR_CT05) * r_hat
                          + lin_interp(alpha, CL_RDA) * da
                          + lin_interp(alpha, CL_RDR) * dr;

            float cn = bilin(alpha, ct, CN_O_CT0, CN_O_CT05)
                     + bilin(alpha, ct, CN_B_CT0, CN_B_CT05) * beta
                     + bilin(alpha, ct, CN_P_CT0, CN_P_CT05) * p_hat
                     + bilin(alpha, ct, CN_R_CT0, CN_R_CT05) * r_hat
                     + lin_interp(alpha, CN_DA) * da
                     + bilin(alpha, ct, CN_DR_CT0, CN_DR_CT05) * dr;

            // Forces and moments
            float qS = 0.5f * RHO * S * vt * vt;
            float L  = qS * cl_lift;
            float D  = qS * cd;
            float Y  = qS * cy;
            float My = qS * CHORD * cm;
            float Lr = qS * SPAN  * cl_roll;
            float Nr = qS * SPAN  * cn;

            float cos_g = cosf(gamma);
            float sin_g = sinf(gamma);
            float cos_a = cosf(alpha);
            float sin_a = sinf(alpha);
            float cos_b = cosf(beta);
            float sin_b = sinf(beta);
            float cos_m = cosf(mu);
            float sin_m = sinf(mu);
            float tan_g = tanf(gamma);
            float tan_b = tanf(beta);
            float sec_b = 1.0f / fmaxf(cos_b, 1e-3f);

            // Stability-axis EOM (Stengel/Phillips)
            *v_dot = ((-D * cos_b + Y * sin_b) / MASS - G * sin_g) / v_stall;
            *g_dot = (L * cos_m - Y * sin_m * cos_b) / (MASS * vt) - (G / vt) * cos_g;
            *a_dot = q - tan_b * (p * cos_a + r * sin_a)
                   + (-L + MASS * G * cos_g * cos_m) / (MASS * vt * fmaxf(cos_b, 1e-3f));
            *b_dot = p * sin_a - r * cos_a
                   + (Y * cos_b + MASS * G * cos_g * sin_m) / (MASS * vt);
            *m_dot = sec_b * (p * cos_a + r * sin_a) + sin_m * tan_g * L / (MASS * vt);

            // Body-axis rotational dynamics (I_xz = 0)
            *p_dot = (I_YY - I_ZZ) / I_XX * q * r + Lr / I_XX;
            *q_dot = (I_ZZ - I_XX) / I_YY * p * r + My / I_YY - (ip_n / I_YY) * r;
            *r_dot = (I_XX - I_YY) / I_ZZ * p * q + Nr / I_ZZ + (ip_n / I_ZZ) * q;
        }

        __device__ void rk4_step8(
            float& gamma, float& vn, float& alpha, float& beta,
            float& mu, float& p, float& q, float& r,
            float de, float da, float throttle, float dr,
            float dt, int n_micro,
            float v_stall, float k_thrust, float& total_reward
        ) {
            total_reward = 0.0f;
            for (int m = 0; m < n_micro; ++m) {
                float k1g,k1v,k1a,k1b,k1m,k1p,k1q,k1r;
                float k2g,k2v,k2a,k2b,k2m,k2p,k2q,k2r;
                float k3g,k3v,k3a,k3b,k3m,k3p,k3q,k3r;
                float k4g,k4v,k4a,k4b,k4m,k4p,k4q,k4r;

                derivs8(gamma, vn, alpha, beta, mu, p, q, r,
                        de, da, throttle, dr, v_stall, k_thrust,
                        &k1g, &k1v, &k1a, &k1b, &k1m, &k1p, &k1q, &k1r);
                derivs8(gamma + 0.5f*dt*k1g, vn + 0.5f*dt*k1v,
                        alpha + 0.5f*dt*k1a, beta + 0.5f*dt*k1b,
                        mu + 0.5f*dt*k1m, p + 0.5f*dt*k1p,
                        q + 0.5f*dt*k1q, r + 0.5f*dt*k1r,
                        de, da, throttle, dr, v_stall, k_thrust,
                        &k2g, &k2v, &k2a, &k2b, &k2m, &k2p, &k2q, &k2r);
                derivs8(gamma + 0.5f*dt*k2g, vn + 0.5f*dt*k2v,
                        alpha + 0.5f*dt*k2a, beta + 0.5f*dt*k2b,
                        mu + 0.5f*dt*k2m, p + 0.5f*dt*k2p,
                        q + 0.5f*dt*k2q, r + 0.5f*dt*k2r,
                        de, da, throttle, dr, v_stall, k_thrust,
                        &k3g, &k3v, &k3a, &k3b, &k3m, &k3p, &k3q, &k3r);
                derivs8(gamma + dt*k3g, vn + dt*k3v,
                        alpha + dt*k3a, beta + dt*k3b,
                        mu + dt*k3m, p + dt*k3p,
                        q + dt*k3q, r + dt*k3r,
                        de, da, throttle, dr, v_stall, k_thrust,
                        &k4g, &k4v, &k4a, &k4b, &k4m, &k4p, &k4q, &k4r);

                gamma += (dt / 6.0f) * (k1g + 2.0f*k2g + 2.0f*k3g + k4g);
                vn    += (dt / 6.0f) * (k1v + 2.0f*k2v + 2.0f*k3v + k4v);
                alpha += (dt / 6.0f) * (k1a + 2.0f*k2a + 2.0f*k3a + k4a);
                beta  += (dt / 6.0f) * (k1b + 2.0f*k2b + 2.0f*k3b + k4b);
                mu    += (dt / 6.0f) * (k1m + 2.0f*k2m + 2.0f*k3m + k4m);
                p     += (dt / 6.0f) * (k1p + 2.0f*k2p + 2.0f*k3p + k4p);
                q     += (dt / 6.0f) * (k1q + 2.0f*k2q + 2.0f*k3q + k4q);
                r     += (dt / 6.0f) * (k1r + 2.0f*k2r + 2.0f*k3r + k4r);

                // === Markov rewards ===
                total_reward += dt * vn * v_stall * sinf(gamma);
                if (gamma >= 0.0f) { break; }

                total_reward -= W_Q_PENALTY * (q * q) * dt;
                if (alpha > 0.24435f) {
                    total_reward -= W_ALPHA_BARRIER_POS * (alpha - 0.24435f) * dt;
                } else if (alpha < -0.17453f) {
                    total_reward -= W_ALPHA_BARRIER_NEG * (-alpha - 0.17453f) * dt;
                }
                total_reward -= W_P_PENALTY * (p * p) * dt;
                total_reward -= W_R_PENALTY * (r * r) * dt;
                total_reward -= W_BETA_PENALTY * (beta * beta) * dt;

                float abs_mu = fabsf(mu);
                if (abs_mu > MU_BARRIER) {
                    float excess = abs_mu - MU_BARRIER;
                    total_reward -= W_MU_BARRIER * excess * excess * dt;
                }

                if (alpha >= ALPHA_HI || alpha <= ALPHA_LO || gamma <= GAMMA_LO
                    || abs_mu >= MU_CRASH || fabsf(p) >= P_CRASH
                    || fabsf(r) >= R_CRASH || fabsf(beta) >= BETA_CRASH) {
                    total_reward -= W_CRASH_PENALTY * v_stall;
                    break;
                }
            }
            // Per-action effort
            total_reward -= W_CONTROL_EFFORT * (de * de) * (dt * n_micro);
            total_reward -= W_AILERON_EFFORT * (da * da) * (dt * n_micro);
            total_reward -= W_RUDDER_EFFORT  * (dr * dr) * (dt * n_micro);
            total_reward += W_THROTTLE_BONUS * throttle * fmaxf(1.0f - vn, 0.0f) * (dt * n_micro);
        }

        // ====================================================================
        // 8D Barycentric (256 corners)
        // ====================================================================
        __device__ void barycentric8(
            float s0, float s1, float s2, float s3,
            float s4, float s5, float s6, float s7,
            const float* b_low, const float* b_high,
            const int* g_shape, const int* strides,
            int* idxs, float* wgts
        ) {
            float n[8]; int i[8]; float d[8];
            float s[8] = {s0, s1, s2, s3, s4, s5, s6, s7};

            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                n[k] = (s[k] - b_low[k]) / (b_high[k] - b_low[k]) * (g_shape[k] - 1);
                n[k] = fmaxf(0.0f, fminf(n[k], (float)(g_shape[k] - 1)));
                i[k] = (int)n[k];
                if (i[k] == g_shape[k] - 1) i[k]--;
                d[k] = n[k] - i[k];
            }

            #pragma unroll
            for (int corner = 0; corner < 256; ++corner) {
                int b0 = (corner >> 7) & 1;
                int b1 = (corner >> 6) & 1;
                int b2 = (corner >> 5) & 1;
                int b3 = (corner >> 4) & 1;
                int b4 = (corner >> 3) & 1;
                int b5 = (corner >> 2) & 1;
                int b6 = (corner >> 1) & 1;
                int b7 =  corner       & 1;

                idxs[corner] = (i[0] + b0) * strides[0] + (i[1] + b1) * strides[1]
                             + (i[2] + b2) * strides[2] + (i[3] + b3) * strides[3]
                             + (i[4] + b4) * strides[4] + (i[5] + b5) * strides[5]
                             + (i[6] + b6) * strides[6] + (i[7] + b7) * strides[7];

                wgts[corner] = (b0 ? d[0] : (1.0f - d[0]))
                             * (b1 ? d[1] : (1.0f - d[1]))
                             * (b2 ? d[2] : (1.0f - d[2]))
                             * (b3 ? d[3] : (1.0f - d[3]))
                             * (b4 ? d[4] : (1.0f - d[4]))
                             * (b5 ? d[5] : (1.0f - d[5]))
                             * (b6 ? d[6] : (1.0f - d[6]))
                             * (b7 ? d[7] : (1.0f - d[7]));
            }
        }

        __global__ void policy_eval_kernel(
            const float* states, const float* actions, const int* policy,
            const float* V, float* new_V, const bool* is_term,
            const float* b_low, const float* b_high,
            const int* g_shape, const int* strides,
            int n_states, float gamma_discount, float dt,
            float v_stall, float k_thrust
        ) {
            int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (s_idx >= n_states) return;
            if (is_term[s_idx]) { new_V[s_idx] = V[s_idx]; return; }

            int a_idx = policy[s_idx];
            float gamma = states[s_idx * 8 + 0];
            float vn    = states[s_idx * 8 + 1];
            float alpha = states[s_idx * 8 + 2];
            float beta  = states[s_idx * 8 + 3];
            float mu    = states[s_idx * 8 + 4];
            float p     = states[s_idx * 8 + 5];
            float q     = states[s_idx * 8 + 6];
            float r     = states[s_idx * 8 + 7];
            float de       = actions[a_idx * 4 + 0];
            float da       = actions[a_idx * 4 + 1];
            float throttle = actions[a_idx * 4 + 2];
            float dr       = actions[a_idx * 4 + 3];
            float reward;

            rk4_step8(gamma, vn, alpha, beta, mu, p, q, r,
                      de, da, throttle, dr, dt, 1,
                      v_stall, k_thrust, reward);

            int idxs[256]; float wgts[256];
            barycentric8(gamma, vn, alpha, beta, mu, p, q, r,
                         b_low, b_high, g_shape, strides, idxs, wgts);

            float expected_v = 0.0f;
            #pragma unroll 16
            for (int i = 0; i < 256; ++i) {
                expected_v = fmaf(wgts[i], V[idxs[i]], expected_v);
            }
            new_V[s_idx] = reward + gamma_discount * expected_v;
        }

        __global__ void policy_improve_kernel(
            const float* states, const float* actions, int* policy,
            const float* V, const bool* is_term,
            const float* b_low, const float* b_high,
            const int* g_shape, const int* strides,
            int n_states, int n_actions, float gamma_discount, float dt,
            float v_stall, float k_thrust, int* policy_changes
        ) {
            int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (s_idx >= n_states) return;
            if (is_term[s_idx]) return;

            float ig = states[s_idx * 8 + 0];
            float iv = states[s_idx * 8 + 1];
            float ia = states[s_idx * 8 + 2];
            float ib = states[s_idx * 8 + 3];
            float im = states[s_idx * 8 + 4];
            float ip = states[s_idx * 8 + 5];
            float iq = states[s_idx * 8 + 6];
            float ir = states[s_idx * 8 + 7];

            float max_q_val = -1e9f;
            int best_a = 0;

            for (int a = 0; a < n_actions; ++a) {
                float gamma = ig, vn = iv, alpha = ia, beta = ib;
                float mu = im, p = ip, q = iq, r = ir;
                float de       = actions[a * 4 + 0];
                float da       = actions[a * 4 + 1];
                float throttle = actions[a * 4 + 2];
                float dr       = actions[a * 4 + 3];
                float reward;

                rk4_step8(gamma, vn, alpha, beta, mu, p, q, r,
                          de, da, throttle, dr, dt, 1,
                          v_stall, k_thrust, reward);

                int idxs[256]; float wgts[256];
                barycentric8(gamma, vn, alpha, beta, mu, p, q, r,
                             b_low, b_high, g_shape, strides, idxs, wgts);

                float expected_v = 0.0f;
                #pragma unroll 16
                for (int i = 0; i < 256; ++i) {
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
        # 8D barycentric needs idxs[256]+wgts[256] = 2 KB local mem per thread.
        # Lower threads/block to manage register pressure.
        self.threads_per_block = 64
        self.blocks_per_grid = (
            (self.n_states + self.threads_per_block - 1) // self.threads_per_block
        )

    def _pull_tensors_from_gpu(self) -> None:
        logger.info("Pulling matrices from VRAM...")
        for attr in ['d_new_value_function', 'd_terminal_mask',
                     'd_states', 'd_actions',
                     'd_bounds_low', 'd_bounds_high',
                     'd_grid_shape', 'd_strides']:
            if hasattr(self, attr):
                delattr(self, attr)
        cp.get_default_memory_pool().free_all_blocks()

        self.value_function = np.empty(self.n_states, dtype=np.float32)
        self.policy = np.empty(self.n_states, dtype=np.int32)

        chunk_size = 5_000_000
        for i in range(0, self.n_states, chunk_size):
            end = min(i + chunk_size, self.n_states)
            self.d_value_function[i:end].get(out=self.value_function[i:end])
            self.d_policy[i:end].get(out=self.policy[i:end])

        del self.d_value_function, self.d_policy
        cp.get_default_memory_pool().free_all_blocks()

    def policy_evaluation(self) -> float:
        delta = float("inf")
        SYNC_INTERVAL = 25
        for i in range(self.config.maximum_iterations):
            self.eval_kernel(
                (self.blocks_per_grid,), (self.threads_per_block,),
                (
                    self.d_states, self.d_actions, self.d_policy,
                    self.d_value_function, self.d_new_value_function,
                    self.d_terminal_mask,
                    self.d_bounds_low, self.d_bounds_high,
                    self.d_grid_shape, self.d_strides,
                    np.int32(self.n_states), np.float32(self.config.gamma),
                    np.float32(self.dt), np.float32(self.v_stall),
                    np.float32(self.k_thrust),
                ),
            )
            d_delta = max_abs_diff_kernel(self.d_new_value_function, self.d_value_function)
            self.d_value_function, self.d_new_value_function = (
                self.d_new_value_function, self.d_value_function
            )
            if i % SYNC_INTERVAL == 0 or i == self.config.maximum_iterations - 1:
                delta = float(d_delta.get())
                if delta < self.config.theta:
                    logger.success(f"Eval converged at step {i} with Δ={delta:.5e}")
                    return delta
        logger.warning(f"Eval hit max iterations ({self.config.maximum_iterations}) with Δ={delta:.5e}")
        return delta

    def policy_improvement(self) -> bool:
        d_policy_changes = cp.zeros(1, dtype=cp.int32)
        self.improve_kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (
                self.d_states, self.d_actions, self.d_policy,
                self.d_value_function, self.d_terminal_mask,
                self.d_bounds_low, self.d_bounds_high,
                self.d_grid_shape, self.d_strides,
                np.int32(self.n_states), np.int32(self.n_actions),
                np.float32(self.config.gamma), np.float32(self.env.airplane.TIME_STEP),
                np.float32(self.env.airplane.STALL_AIRSPEED),
                np.float32(self.env.airplane.THROTTLE_LINEAR_MAPPING),
                d_policy_changes,
            ),
        )
        changes = int(d_policy_changes.get()[0])
        tolerance_threshold = int(self.n_states * 0.0001)
        policy_stable = (changes <= tolerance_threshold)
        if not policy_stable:
            logger.info(
                f"Policy updated: {changes} states changed (tolerance: {tolerance_threshold})"
            )
        return policy_stable

    def run(self) -> None:
        for n in range(self.config.n_steps):
            logger.info(f"--- Iteration {n + 1}/{self.config.n_steps} ---")
            self.policy_evaluation()
            is_stable = self.policy_improvement()
            if is_stable:
                logger.success(f"Algorithm converged at iteration {n + 1}.")
                break
        self._pull_tensors_from_gpu()

    def save(self, filepath: Path | None = None) -> None:
        if filepath is None:
            filepath = Path.cwd() / f"{self.env.unwrapped.__class__.__name__}_policy.npz"
        filepath = filepath.with_suffix(".npz")
        logger.info(f"Saving policy to {filepath.resolve()}...")
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
        logger.success(f"Policy saved.")

    @classmethod
    def load(cls, filepath: Path, env: gym.Env = None) -> "PolicyIterationSpin":
        filepath = filepath.with_suffix(".npz")
        logger.info(f"Loading policy from {filepath.resolve()}...")
        data = np.load(filepath)

        instance = cls.__new__(cls)
        instance.env = env
        instance.config = PolicyIterationSpinConfig()
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
        return instance
