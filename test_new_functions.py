import json
import os
from datetime import datetime
from agent_tools import (
    CONSTANTS,
    GenerateGridLayout,
    ConfigureCPEfficiency,
    ApplyMeasuredCompensation,
    ApplyCollimateLens,
    ApplyAxiconPhase,
    ApplyAiryPhase,
    ApplyBeamSteering,
    ApplyVortexPhase,
    ApplyDammannGrating,
    CalculatePBRotation,
    SimulateMetasurface,
    SimulateNearfieldPropagation,
    init_run_dir,
    get_layout_path,
    RESULTS_ROOT,
)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# ============================================================
# 配置: 实测数据文件路径 (根据实际位置修改)
# ============================================================
DATA_DIR = os.path.abspath(os.path.dirname(__file__))

LCP_PHASE_CSV = os.path.join(DATA_DIR, "Left_circular_Phase_mm.csv")
LCP_MAG_CSV = os.path.join(DATA_DIR, "Left_circular_Magnitude_mm.csv")
RCP_PHASE_CSV = os.path.join(DATA_DIR, "Right_circular_Phase_mm.csv")
RCP_MAG_CSV = os.path.join(DATA_DIR, "Right_circular_Magnitude_mm.csv")

FEED_DISTANCE = 0.240


def init_grid(radius=0.13):
    """初始化六边形阵列"""
    res = GenerateGridLayout().call(
        json.dumps({"layout_type": "hex", "radius": radius})
    )
    print(f"   {res}")


def configure_cp(eta=0.85, pol="LCP"):
    """配置圆极化参数"""
    res = ConfigureCPEfficiency().call(json.dumps({"eta": eta, "incident_pol": pol}))
    print(f"   {res}")


def pb_and_sim(test_name):
    """PB转角计算 + 仿真"""
    res_pb = CalculatePBRotation().call(json.dumps({}))
    print(f"   {res_pb}")
    res_sim = SimulateMetasurface().call(json.dumps({}))
    print(f"   {res_sim}")


# ============================================================
# 测试 A: LCP 入射 → 平面波出射 (实测数据补偿)
# ============================================================
def test_lcp_collimate(eta=0.85):
    test_name = "LCP_collimate"
    init_run_dir(test_name)

    print(f"\n{'=' * 60}")
    print(f"  Test A: {test_name} (LCP feed → plane wave)")
    print(f"{'=' * 60}")

    print(">>> Step 1: Init grid")
    init_grid()

    print(">>> Step 2: Configure CP (LCP)")
    configure_cp(eta=eta, pol="LCP")

    print(">>> Step 3: Load measured LCP data & compute compensation")
    res = ApplyMeasuredCompensation().call(
        json.dumps({"phase_csv": LCP_PHASE_CSV, "magnitude_csv": LCP_MAG_CSV})
    )
    print(f"   {res}")

    print(">>> Step 4: PB rotation & Simulation")
    pb_and_sim(test_name)
    print(f"\n>>> Test A [{test_name}] COMPLETE.\n")


# ============================================================
# 测试 H: 仅生成 Airy 波束（理论公式，无实测补偿）
# ============================================================
def test_airy_only(eta=0.85, coeff=5e6):
    test_name = "Airy_only"
    init_run_dir(test_name)

    print(f"\n{'=' * 60}")
    print(f"  Test H: {test_name} (theoretical Airy beam)")
    print(f"{'=' * 60}")

    print(">>> Step 1: Init grid")
    init_grid()

    print(">>> Step 2: Configure CP (LCP)")
    configure_cp(eta=eta, pol="LCP")

    print(f">>> Step 3: Apply Airy cubic phase (a3={coeff:.1e})")
    res = ApplyAiryPhase().call(
        json.dumps({"cubic_coeff": coeff, "separable": True, "mode": "overwrite"})
    )
    print(f"   {res}")

    print(">>> Step 3b: Near-field propagation (Airy trajectory)")
    res_nf = SimulateNearfieldPropagation().call(json.dumps({
        "z_max": 0.8,
        "z_steps": 80,
        "grid_size": 256,
        "test_title": f"Airy (a3={coeff:.1e})"
    }))
    print(f"   {res_nf}")

    print(">>> Step 4: PB rotation & Simulation")
    pb_and_sim(test_name)
    print(f"\n>>> Test H [{test_name}] COMPLETE.\n")


# ============================================================
# 测试 I: 仅生成 Bessel 波束（理论公式，无实测补偿）
# ============================================================
def test_bessel_only(eta=0.85, cone_angle_deg=60):
    test_name = f"Bessel_{cone_angle_deg}deg"
    init_run_dir(test_name)

    print(f"\n{'=' * 60}")
    print(f"  Test I: {test_name} (theoretical Bessel beam)")
    print(f"{'=' * 60}")

    print(">>> Step 1: Init grid")
    init_grid(radius=0.13)

    print(">>> Step 2: Configure CP (LCP)")
    configure_cp(eta=eta, pol="LCP")

    print(f">>> Step 3: Apply axicon phase (cone={cone_angle_deg}°)")
    res = ApplyAxiconPhase().call(
        json.dumps({"cone_angle_deg": cone_angle_deg, "mode": "overwrite"})
    )
    print(f"   {res}")

    print(">>> Step 4: PB rotation")
    res_pb = CalculatePBRotation().call(json.dumps({}))
    print(f"   {res_pb}")

    print(">>> Step 5: Near-field propagation (Bessel verification)")
    res_nf = SimulateNearfieldPropagation().call(json.dumps({
        "z_max": 0.20,
        "z_steps": 100,
        "grid_size": 256,
        "n_xoy_slices": 21,
        "test_title": f"Bessel (cone={cone_angle_deg}°)"
    }))
    print(f"   {res_nf}")

    print(">>> Step 6: Far-field simulation")
    pb_and_sim(test_name)
    print(f"\n>>> Test I [{test_name}] COMPLETE.\n")


# ============================================================
# 测试 F: LCP 入射 → 补偿 + 达曼光栅 (多波束)
# ============================================================
def test_lcp_collimate_dammann(eta=0.85, beam_order=3):
    test_name = f"LCP_collimate_dammann_{beam_order}x{beam_order}"
    init_run_dir(test_name)

    print(f"\n{'=' * 60}")
    print(f"  Test F: {test_name}")
    print(f"{'=' * 60}")

    print(">>> Step 1: Init grid")
    init_grid()

    print(">>> Step 2: Configure CP (LCP)")
    configure_cp(eta=eta, pol="LCP")

    print(">>> Step 3: Load LCP data & compensate")
    res = ApplyMeasuredCompensation().call(
        json.dumps({"phase_csv": LCP_PHASE_CSV, "magnitude_csv": LCP_MAG_CSV})
    )
    print(f"   {res}")

    print(f">>> Step 4: Add {beam_order}x{beam_order} Dammann grating")
    res = ApplyDammannGrating().call(
        json.dumps({"beam_order": beam_order, "period_multiple": 12})
    )
    print(f"   {res}")

    print(">>> Step 5: PB rotation & Simulation")
    pb_and_sim(test_name)
    print(f"\n>>> Test F [{test_name}] COMPLETE.\n")


# ============================================================
# 测试 G: 对比 — 理论公式准直 vs 实测数据补偿
# ============================================================
def test_compare_theoretical_vs_measured(eta=0.85):
    """对比理论公式和实测补偿的远场差异"""

    # G1: 理论准直
    name_theo = "LCP_theoretical_collimate"
    init_run_dir(name_theo)

    print(f"\n{'=' * 60}")
    print(f"  Test G1: {name_theo} (theoretical formula)")
    print(f"{'=' * 60}")

    init_grid()
    configure_cp(eta=eta, pol="LCP")

    res = ApplyCollimateLens().call(
        json.dumps({"focal_length": FEED_DISTANCE, "mode": "overwrite"})
    )
    print(f"   {res}")

    import pandas as pd_local

    layout_path = get_layout_path()
    df = pd_local.read_csv(layout_path)
    if os.path.exists(LCP_MAG_CSV):
        from agent_tools import parse_measured_csv, interpolate_to_hex_grid

        xm, ym, mag_2d = parse_measured_csv(LCP_MAG_CSV)
        measured_mag = interpolate_to_hex_grid(
            xm, ym, mag_2d, df["x"].values, df["y"].values
        )
        import numpy as np
        df["measured_mag"] = np.clip(measured_mag, 0, None)
        df["mag"] = df["measured_mag"]
        df.to_csv(layout_path, index=False)

    pb_and_sim(name_theo)

    # G2: 实测补偿 (已在 Test A 中做过，这里再做一次方便对比)
    test_lcp_collimate(eta=eta)

    print("\n>>> Compare result_LCP_theoretical_collimate_farfield.png")
    print(">>>     vs  result_LCP_collimate_farfield.png")
    print(">>> Measured compensation should show a tighter broadside beam.\n")

if __name__ == "__main__":
    print("=" * 60)
    print("  Metasurface Test Suite — Measured Feed Compensation")
    print(f"  Feed distance: {FEED_DISTANCE * 1000:.0f} mm")
    print(f"  Working freq: {CONSTANTS['freq'] / 1e9:.1f} GHz")
    print(f"  Unit cell: {CONSTANTS['p'] * 1000:.1f} mm")
    print("=" * 60)

    test_bessel_only(eta=0.85, cone_angle_deg=37)

    # # Airy波束 - 合理coeff
    # test_airy_only(coeff=8.6e4)

    print("\n" + "=" * 60)
    print(f"  All tests complete. Check {RESULTS_ROOT}/ directory.")
    print("=" * 60)