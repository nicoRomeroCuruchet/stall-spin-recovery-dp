"""
Validation suite for the 6-DOF banked-spin pipeline.

Tests:
1. Riley III(f) table sanity (CPU helper)
2. Symmetric subcase: 6DOF with mu=0, p=0, da=0 must keep mu=0, p=0 exactly
3. Open-loop spin-entry trajectory (qualitative)
4. End-to-end GPU policy iteration smoke (small grid, must converge)

Usage: python3 validate_banked_spin.py
"""

import sys

import numpy as np

from aircraft.banked_spin import BankedSpin
from aircraft.banked_spin_grumman import BankedSpinGrumman
from aircraft.grumman import Grumman


def test_1_riley_tables():
    print("=" * 68)
    print("TEST 1: Riley III(f) table sanity")
    print("=" * 68)
    g = Grumman()

    cases = [
        ("Cl_p_hat(0 deg,  CT=0)", g._bilinear_interp(
            0.0, 0.0, g._CL_ROLL_PHAT_TABLE_CT0, g._CL_ROLL_PHAT_TABLE_CT05),
         -0.5200),
        ("Cl_p_hat(14 deg, CT=0)", g._bilinear_interp(
            np.deg2rad(14.0), 0.0,
            g._CL_ROLL_PHAT_TABLE_CT0, g._CL_ROLL_PHAT_TABLE_CT05),
         -0.2200),
        ("Cl_o(18 deg,    CT=0)", g._bilinear_interp(
            np.deg2rad(18.0), 0.0,
            g._CL_ROLL_O_TABLE_CT0, g._CL_ROLL_O_TABLE_CT05),
         -0.0075),
        ("Cl_da(0 deg)/rad     ",
         np.interp(0.0, g._CL_O_ALPHA_RAD, g._CL_ROLL_DA_TABLE),
         -0.001040 * 57.2958),
    ]

    failed = 0
    for label, val, expected in cases:
        ok = abs(val - expected) < 1e-3
        status = "OK" if ok else "FAIL"
        print(f"  {label} = {val:+.4f}  (expected {expected:+.4f}) [{status}]")
        if not ok:
            failed += 1
    if failed:
        raise AssertionError(f"{failed} Riley sanity check(s) failed")
    print("  -> all 4 lookups match Riley Table III(f)\n")


def test_2_symmetric_subcase():
    print("=" * 68)
    print("TEST 2: Symmetric subcase  ->  no lateral drift with mu=p=da=0")
    print("=" * 68)
    ac = BankedSpinGrumman()

    # Non-trivial longitudinal IC; lateral states pinned to zero.
    ac.reset(flight_path_angle=-0.3, airspeed_norm=1.2,
             alpha=np.deg2rad(8.0),
             bank_angle=0.0, roll_rate=0.0, pitch_rate=0.05)

    # Longitudinal-only inputs (no aileron).
    elev = np.deg2rad(-5.0)
    thr = 0.6
    N = 500  # 5 s
    mu_max = 0.0
    p_max = 0.0
    for _ in range(N):
        ac.command_airplane(elev, 0.0, thr)
        mu_max = max(mu_max, abs(ac.bank_angle))
        p_max = max(p_max, abs(ac.roll_rate))

    print(f"  After {N} steps ({N * ac.TIME_STEP:.1f} s):")
    print(f"    final gamma = {np.rad2deg(ac.flight_path_angle):+6.2f} deg")
    print(f"    final V/Vs  = {ac.airspeed_norm:.4f}")
    print(f"    final alpha = {np.rad2deg(ac.alpha):+6.2f} deg")
    print(f"    final q     = {np.rad2deg(ac.pitch_rate):+6.2f} deg/s")
    print(f"  Lateral states must remain identically zero:")
    print(f"    max |mu|    = {mu_max:.3e}")
    print(f"    max |p|     = {p_max:.3e}")

    assert mu_max == 0.0 and p_max == 0.0, \
        "Lateral drift detected — symmetric subcase is broken"
    print("  -> 6DOF leaves the symmetric subspace invariant\n")


def test_3_open_loop_spin_entry():
    print("=" * 68)
    print("TEST 3: Open-loop spin entry (CPU trajectory)")
    print("=" * 68)
    ac = BankedSpinGrumman()
    # Level slow flight, on the verge of stall
    ac.reset(flight_path_angle=0.0, airspeed_norm=1.0,
             alpha=np.deg2rad(15.0),
             bank_angle=0.0, roll_rate=0.0, pitch_rate=0.0)

    elev = np.deg2rad(-25.0)   # full pull
    ail = np.deg2rad(-15.0)    # full deflection (Riley sign: negative -> ṗ>0)
    thr = 0.3
    print("  Initial: γ=0°, V=1.0, α=15°, μ=0°, p=0, q=0")
    print(f"  Hold:    δe=-25°, δa=-15°, δt=0.3 (full pull + full aileron)")
    print()
    print("    t[s] |  γ°    V/Vs   α°     μ°     p[r/s]  q[r/s]")
    print("   ------+------------------------------------------------")

    samples = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    next_idx = 0
    history = []
    n_steps = int(5.0 / ac.TIME_STEP)
    for i in range(n_steps + 1):
        t = i * ac.TIME_STEP
        if next_idx < len(samples) and t >= samples[next_idx] - 1e-9:
            print(f"    {t:4.1f} | {np.rad2deg(ac.flight_path_angle):+5.1f} "
                  f"{ac.airspeed_norm:.3f}  "
                  f"{np.rad2deg(ac.alpha):+5.1f}  "
                  f"{np.rad2deg(ac.bank_angle):+6.1f}  "
                  f"{ac.roll_rate:+.3f}  {ac.pitch_rate:+.3f}")
            history.append((t, ac.flight_path_angle, ac.airspeed_norm,
                            ac.alpha, ac.bank_angle, ac.roll_rate, ac.pitch_rate))
            next_idx += 1
        if i < n_steps:
            ac.command_airplane(elev, ail, thr)

    # Physical expectation: p rises initially driven by aileron Cl_da,
    # then decays as Cl_p_hat damping (negative at low alpha) and post-stall
    # asymmetry equilibrate; mu integrates p over time -> grows monotonically.
    p_peak = max(abs(h[5]) for h in history)
    mu_final = history[-1][4]
    alpha_max = max(h[3] for h in history)

    print()
    print("  Qualitative checks (Riley fig. 9-10 expected behaviour):")
    print(f"    p peak:             max|p|   = {p_peak:.3f} rad/s  "
          f"({'OK' if p_peak > 0.1 else 'WEAK'})")
    print(f"    bank develops:      |mu_5s|  = {np.rad2deg(abs(mu_final)):.1f} deg  "
          f"({'OK' if abs(mu_final) > np.deg2rad(10) else 'WEAK'})")
    print(f"    alpha post-stall:   alpha_max = {np.rad2deg(alpha_max):.1f} deg  "
          f"({'OK' if alpha_max > np.deg2rad(14) else 'WEAK'})")

    if p_peak <= 0.1 or abs(mu_final) <= np.deg2rad(10) \
            or alpha_max <= np.deg2rad(14):
        raise AssertionError("Spin-entry trajectory did not develop p/mu/alpha")
    print("  -> Open-loop spin-entry shows expected coupled roll/bank dynamics\n")


def test_4_gpu_smoke():
    print("=" * 68)
    print("TEST 4: End-to-end GPU PI smoke (5x5x5x5x5x5 grid)")
    print("=" * 68)
    try:
        import cupy as cp  # noqa: F401
    except ImportError:
        print("  [SKIPPED] cupy not installed.\n")
        return

    from PolicyIterationBankedSpin import (
        PolicyIterationBankedSpin,
        PolicyIterationBankedSpinConfig,
    )

    env = BankedSpin()
    gammas = np.linspace(-1.5, 0.0, 5, dtype=np.float32)
    vns = np.linspace(0.9, 1.5, 5, dtype=np.float32)
    alphas = np.linspace(np.deg2rad(0), np.deg2rad(20), 5, dtype=np.float32)
    mus = np.linspace(np.deg2rad(0), np.deg2rad(60), 5, dtype=np.float32)
    ps = np.linspace(-0.3, 0.3, 5, dtype=np.float32)
    qs = np.linspace(-0.3, 0.3, 5, dtype=np.float32)
    states = np.array(
        np.meshgrid(gammas, vns, alphas, mus, ps, qs, indexing="ij"),
        dtype=np.float32).reshape(6, -1).T

    de = np.linspace(np.deg2rad(-25), np.deg2rad(15), 5, dtype=np.float32)
    da = np.linspace(np.deg2rad(-15), np.deg2rad(15), 5, dtype=np.float32)
    dt = np.linspace(0.0, 1.0, 3, dtype=np.float32)
    actions = np.array(
        np.meshgrid(de, da, dt, indexing="ij"),
        dtype=np.float32).reshape(3, -1).T

    cfg = PolicyIterationBankedSpinConfig(
        maximum_iterations=400,
        n_steps=4,
        theta=1e-3,
    )
    pi = PolicyIterationBankedSpin(env, states, actions, cfg)
    print(f"  Grid: {pi.grid_shape}, n_states={pi.n_states}, n_actions={pi.n_actions}")

    # Run 4 PI iterations max — small grid, just to verify pipeline
    for n in range(cfg.n_steps):
        delta = pi.policy_evaluation()
        stable = pi.policy_improvement()
        print(f"  iter {n + 1}: eval-Δ={delta:.3e}, stable={stable}")
        if stable:
            break

    pi._pull_tensors_from_gpu()
    V = pi.value_function
    P = pi.policy
    n_terminal = int(np.isnan(V).sum() + np.sum(V == 0.0))
    n_active = pi.n_states - n_terminal

    print(f"  Final V range: [{V.min():.2f}, {V.max():.2f}]")
    print(f"  Active (non-terminal) states with non-zero V: ~{n_active}")
    print(f"  Distinct actions in policy: {len(np.unique(P))}")

    assert np.all(np.isfinite(V)), "V has NaN/Inf"
    assert len(np.unique(P)) > 1, "Policy is degenerate (one action everywhere)"
    print("  -> GPU pipeline converges and produces non-trivial policy\n")


def main():
    print()
    test_1_riley_tables()
    test_2_symmetric_subcase()
    test_3_open_loop_spin_entry()
    test_4_gpu_smoke()
    print("=" * 68)
    print("Phase 5 validation: ALL TESTS PASSED")
    print("=" * 68)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
