"""
PPO-SymmetricStall.py — PPO Training for Symmetric Stall Recovery
Uses Stable Baselines 3 with CUDA GPU acceleration.
Saves best_model.zip to policy_symmetric_stall/models/ for use with PPO_vs_PI.py.
"""
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from aircraft.symmetric_stall import SymmetricStall

os.environ["NUMBA_THREADING_LAYER"] = "omp"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Observation bounds (physical limits of the state space) ───────────────────
OBS_LOW  = np.array([-np.pi,       0.3,  np.deg2rad(-50), np.deg2rad(-70)], dtype=np.float32)
OBS_HIGH = np.array([ np.pi / 4,   2.5,  np.deg2rad( 50), np.deg2rad( 70)], dtype=np.float32)

# ── Training hyperparameters ──────────────────────────────────────────────────
N_ENVS          = 16
N_STEPS         = 2048
BATCH_SIZE      = 512
N_EPOCHS        = 10
LEARNING_RATE   = 3e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
ENT_COEF        = 0.005
TOTAL_TIMESTEPS = 10_000_000
POLICY_KWARGS   = dict(net_arch=[256, 256])

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("policy_symmetric_stall/models")
LOG_DIR   = Path("policy_symmetric_stall/logs")

# ── Evaluation initial conditions (match main.py) ─────────────────────────────
GAMMA_0_DEG = -30.0
V_NORM_0    = 0.95
ALPHA_0_DEG = 20.0
Q_0_DEG     = 0.0
MAX_STEPS         = 1500
MAX_EPISODE_STEPS = 1500   # 15 s de simulación (TIME_STEP=0.01)


# ── Environment wrapper ───────────────────────────────────────────────────────

class SymmetricStallEnv(SymmetricStall):
    """Adds observation_space for SB3 compatibility. No other changes."""
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, dtype=np.float32
        )


def _make_env():
    def _init():
        return TimeLimit(SymmetricStallEnv(), max_episode_steps=MAX_EPISODE_STEPS)
    return _init


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> PPO:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"[*] Spawning {N_ENVS} parallel environments...")
    vec_env  = VecNormalize(
        SubprocVecEnv([_make_env() for _ in range(N_ENVS)]),
        norm_obs=False, norm_reward=True, clip_reward=10.0,
    )
    eval_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(TimeLimit(SymmetricStallEnv(), max_episode_steps=MAX_EPISODE_STEPS))]),
        norm_obs=False, norm_reward=True, clip_reward=10.0, training=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=30,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1_000_000 // N_ENVS, 1),   # cada ~1M timesteps globales
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="ppo",
        verbose=1,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    logger.info("[*] Building PPO model on CUDA GPU...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        policy_kwargs=POLICY_KWARGS,
        device="cpu",
        verbose=1,
    )

    logger.info(f"[*] Training for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    vec_env.save(MODEL_DIR / "vecnormalize.pkl")
    vec_env.close()
    logger.info(f"[+] Best model saved to: {(MODEL_DIR / 'best_model.zip').resolve()}")
    return model


# ── Trajectory simulation ─────────────────────────────────────────────────────

def simulate_trajectory(model: PPO) -> dict:
    """Rolls out the trained policy from fixed initial conditions."""
    env     = SymmetricStallEnv()
    v_stall = env.airplane.STALL_AIRSPEED
    step_dt = env.airplane.TIME_STEP

    obs, _ = env.specific_reset(
        np.deg2rad(GAMMA_0_DEG), V_NORM_0,
        np.deg2rad(ALPHA_0_DEG), np.deg2rad(Q_0_DEG),
    )

    t, h      = 0.0, 0.0
    has_dived = False
    hist      = {k: [] for k in ("t", "gamma", "v", "alpha", "q", "de", "dt", "h")}

    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)

        hist["t"].append(t)
        hist["gamma"].append(np.rad2deg(obs[0]))
        hist["v"].append(obs[1])
        hist["alpha"].append(np.rad2deg(obs[2]))
        hist["q"].append(np.rad2deg(obs[3]))
        hist["de"].append(np.rad2deg(action[0]))
        hist["dt"].append(action[1])
        hist["h"].append(h)

        obs, _, terminated, _, _ = env.step(action)
        v_true = obs[1] * v_stall
        h     += v_true * np.sin(obs[0]) * step_dt
        t     += step_dt

        new_gamma = np.rad2deg(obs[0])
        if new_gamma < -2.0:
            has_dived = True

        if has_dived and new_gamma >= 0.0:
            hist["t"].append(t)
            hist["gamma"].append(new_gamma)
            hist["v"].append(obs[1])
            hist["alpha"].append(np.rad2deg(obs[2]))
            hist["q"].append(np.rad2deg(obs[3]))
            hist["de"].append(hist["de"][-1])
            hist["dt"].append(hist["dt"][-1])
            hist["h"].append(h)
            logger.info(f"[+] Recovery at {t:.2f}s | altitude loss: {h:.2f} m")
            break

        if terminated:
            logger.warning(f"[-] Episode terminated (failure) at {t:.2f}s")
            break

    return {k: np.array(v) for k, v in hist.items()}


# ── Trajectory plot ───────────────────────────────────────────────────────────

def plot_trajectory(data: dict, prefix: str) -> None:
    """7-panel trajectory figure matching the style of main.py."""
    C_PPO  = '#2CA02C'   # green — PPO policy
    C_CTRL = '#E87C1E'   # orange — control signals
    C_ALT  = '#D62728'   # red — altitude loss

    fig, axs = plt.subplots(7, 1, figsize=(8, 16), sharex=True)
    fig.suptitle(r"Symmetric Stall Recovery — PPO Policy", fontsize=14)

    axs[0].plot(data["t"], data["gamma"], color=C_PPO,  linewidth=2, label="PPO")
    axs[0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axs[0].set_ylabel(r'$\gamma$ (deg)')

    axs[1].plot(data["t"], data["v"],     color=C_PPO,  linewidth=2, label="PPO")
    axs[1].set_ylabel(r'$V/V_s$')

    axs[2].plot(data["t"], data["alpha"], color=C_PPO,  linewidth=2, label="PPO")
    axs[2].set_ylabel(r'$\alpha$ (deg)')

    axs[3].plot(data["t"], data["q"],     color=C_PPO,  linewidth=2, label="PPO")
    axs[3].set_ylabel(r'$q$ (deg/s)')

    axs[4].step(data["t"], data["de"],    color=C_CTRL, linewidth=2, where="post", label="PPO")
    axs[4].set_ylabel(r'$\delta_e$ (deg)')

    axs[5].step(data["t"], data["dt"],    color=C_CTRL, linewidth=2, where="post", label="PPO")
    axs[5].set_ylabel(r'$\delta_t$')
    axs[5].set_ylim([-0.05, 1.05])

    axs[6].plot(data["t"], data["h"],     color=C_ALT,  linewidth=2, label="PPO")
    axs[6].set_ylabel('Altitude Loss (m)')
    axs[6].set_xlabel('Time (s)')

    for ax in axs:
        ax.grid(True, linestyle='-', alpha=0.4)
        ax.legend(loc="best")

    plt.tight_layout()
    out_path = Path("results") / f"{prefix}_trajectory.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"[+] Trajectory plot saved to: {out_path.resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    train()

    logger.info("[*] Loading best model for evaluation...")
    best_model = PPO.load(MODEL_DIR / "best_model.zip", device="cpu")

    data = simulate_trajectory(best_model)
    plot_trajectory(data, "PPO_symmetric_stall")


if __name__ == "__main__":
    main()
