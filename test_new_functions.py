import json
import os
import shutil
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
    DEFAULT_LAYOUT_PATH,
    OUTPUT_DIR,
)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
BACKUP_ROOT = os.path.join(PROJECT_ROOT, "outputs_backup")
_current_backup_dir = None

# ============================================================
# 配置: 实测数据文件路径 (根据实际位置修改)
# ============================================================
DATA_DIR = os.path.abspath(os.path.dirname(__file__))

LCP_PHASE_CSV = os.path.join(DATA_DIR, "Left_circular_Phase_mm.csv")
LCP_MAG_CSV = os.path.join(DATA_DIR, "Left_circular_Magnitude_mm.csv")
RCP_PHASE_CSV = os.path.join(DATA_DIR, "Right_circular_Phase_mm.csv")
RCP_MAG_CSV = os.path.join(DATA_DIR, "Right_circular_Magnitude_mm.csv")

FEED_DISTANCE = 0.240


def backup_outputs_auto():
    """备份当前outputs目录和输入csv文件到带时间戳的文件夹"""
    global _current_backup_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _current_backup_dir = os.path.join(BACKUP_ROOT, f"backup_{timestamp}")
    os.makedirs(_current_backup_dir, exist_ok=True)

    for csv_file in [LCP_PHASE_CSV, LCP_MAG_CSV, RCP_PHASE_CSV, RCP_MAG_CSV]:
        if os.path.exists(csv_file):
            shutil.copy(
                csv_file, os.path.join(_current_backup_dir, os.path.basename(csv_file))
            )

    print(f"[AUTO BACKUP] Input CSVs saved to outputs_backup/backup_{timestamp}/")


def restore_latest_backup():
    """恢复最近一次备份到outputs目录"""
    if not os.path.exists(BACKUP_ROOT):
        return False
    backups = sorted([d for d in os.listdir(BACKUP_ROOT) if d.startswith("backup_")])
    if not backups:
        return False
    latest = backups[-1]
    backup_dir = os.path.join(BACKUP_ROOT, latest)
    for f in os.listdir(backup_dir):
        src = os.path.join(backup_dir, f)
        dst = os.path.join(OUTPUT_DIR, f)
        shutil.copy(src, dst)
    print(f"[RESTORE] Restored from {latest}")
    return True


def backup_results(test_name):
    """把仿真结果保存到时间戳备份目录，清理outputs/"""
    global _current_backup_dir
    if _current_backup_dir is None:
        print("[WARNING] No backup directory, skipping save")
        return

    base = DEFAULT_LAYOUT_PATH.replace(".csv", "")
    for sfx in [".csv", "_layout.png", "_farfield.png", "_incident_vs_comp.png"]:
        src = base + sfx if sfx != ".csv" else DEFAULT_LAYOUT_PATH
        dst = os.path.join(_current_backup_dir, f"result_{test_name}{sfx}")
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"   Saved: result_{test_name}{sfx}")

    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("current_layout"):
            os.remove(os.path.join(OUTPUT_DIR, f))
    print(f"[CLEANUP] outputs/ cleared")


def init_grid(radius=0.08):
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
    """PB转角计算 + 仿真 + 备份"""
    res_pb = CalculatePBRotation().call(json.dumps({}))
    print(f"   {res_pb}")
    res_sim = SimulateMetasurface().call(json.dumps({}))
    print(f"   {res_sim}")
    backup_results(test_name)


# ============================================================
# 测试 A: LCP 入射 → 平面波出射 (实测数据补偿)
# ============================================================
def test_lcp_collimate(eta=0.85):
    name = "LCP_collimate"
    print(f"\n{'=' * 60}")
    print(f"  Test A: {name} (LCP feed → plane wave)")
    print(f"{'=' * 60}")

    # 1. 初始化阵列
    print(">>> Step 1: Init grid")
    init_grid()

    # 2. 配置 LCP 入射
    print(">>> Step 2: Configure CP (LCP)")
    configure_cp(eta=eta, pol="LCP")

    # 3. 加载实测数据并计算补偿相位
    print(">>> Step 3: Load measured LCP data & compute compensation")
    res = ApplyMeasuredCompensation().call(
        json.dumps({"phase_csv": LCP_PHASE_CSV, "magnitude_csv": LCP_MAG_CSV})
    )
    print(f"   {res}")

    # 4. PB转角 + 仿真
    print(">>> Step 4: PB rotation & Simulation")
    pb_and_sim(name)
    print(f"\n>>> Test A [{name}] COMPLETE.\n")


# ============================================================
# 测试 B: RCP 入射 → 平面波出射 (实测数据补偿)
# ============================================================
def test_rcp_collimate(eta=0.85):
    name = "RCP_collimate"
    print(f"\n{'=' * 60}")
    print(f"  Test B: {name} (RCP feed → plane wave)")
    print(f"{'=' * 60}")

    print(">>> Step 1: Init grid")
    init_grid()

    print(">>> Step 2: Configure CP (RCP)")
    configure_cp(eta=eta, pol="RCP")

    print(">>> Step 3: Load measured RCP data & compute compensation")
    res = ApplyMeasuredCompensation().call(
        json.dumps({"phase_csv": RCP_PHASE_CSV, "magnitude_csv": RCP_MAG_CSV})
    )
    print(f"   {res}")

    print(">>> Step 4: PB rotation & Simulation")
    pb_and_sim(name)
    print(f"\n>>> Test B [{name}] COMPLETE.\n")


# ============================================================
# 测试 C: LCP 入射 → 补偿 + 波束偏转
# ============================================================
def test_lcp_collimate_steer(eta=0.85, theta=20, phi=0):
    name = f"LCP_collimate_steer_t{theta}_p{phi}"
    print(f"\n{'=' * 60}")
    print(f"  Test C: {name}")
    print(f"{'=' * 60}")

    print(">>> Step 1: Init grid")
    init_grid()

    print(">>> Step 2: Configure CP (LCP)")
    configure_cp(eta=eta, pol="LCP")

    # 先补偿
    print(">>> Step 3: Load LCP data & compensate")
    res = ApplyMeasuredCompensation().call(
        json.dumps({"phase_csv": LCP_PHASE_CSV, "magnitude_csv": LCP_MAG_CSV})
    )
    print(f"   {res}")

    # 再叠加偏转相位
    print(f">>> Step 4: Add beam steering (θ={theta}°, φ={phi}°)")
    res = ApplyBeamSteering().call(
        json.dumps({"steer_theta_deg": theta, "steer_phi_deg": phi, "mode": "add"})
    )
    print(f"   {res}")

    print(">>> Step 5: PB rotation & Simulation")
    pb_and_sim(name)
    print(f"\n>>> Test C [{name}] COMPLETE.\n")


# ============================================================
# 测试 D: LCP 入射 → 补偿 + Bessel 波束
# ============================================================
def test_lcp_collimate_bessel(eta=0.85, cone=10):
    name = f"LCP_collimate_bessel_{cone}deg"
    print(f"\n{'=' * 60}")
    print(f"  Test D: {name}")
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

    print(f">>> Step 4: Add Bessel axicon (cone={cone}°)")
    res = ApplyAxiconPhase().call(json.dumps({"cone_angle_deg": cone, "mode": "add"}))
    print(f"   {res}")

    print(">>> Step 5: PB rotation & Simulation")
    pb_and_sim(name)
    print(f"\n>>> Test D [{name}] COMPLETE.\n")


# ============================================================
# 测试 E: LCP 入射 → 补偿 + Airy 波束
# ============================================================
def test_lcp_collimate_airy(eta=0.85, coeff=5e6):
    name = "LCP_collimate_airy"
    print(f"\n{'=' * 60}")
    print(f"  Test E: {name}")
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

    print(f">>> Step 4: Add Airy cubic phase (a3={coeff:.1e})")
    res = ApplyAiryPhase().call(
        json.dumps({"cubic_coeff": coeff, "separable": True, "mode": "add"})
    )
    print(f"   {res}")

    print(">>> Step 5: PB rotation & Simulation")
    pb_and_sim(name)
    print(f"\n>>> Test E [{name}] COMPLETE.\n")


# ============================================================
# 测试 F: LCP 入射 → 补偿 + 达曼光栅 (多波束)
# ============================================================
def test_lcp_collimate_dammann(eta=0.85, beam_order=3):
    name = f"LCP_collimate_dammann_{beam_order}x{beam_order}"
    print(f"\n{'=' * 60}")
    print(f"  Test F: {name}")
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
    pb_and_sim(name)
    print(f"\n>>> Test F [{name}] COMPLETE.\n")


# ============================================================
# 测试 G: 对比 — 理论公式准直 vs 实测数据补偿
# ============================================================
def test_compare_theoretical_vs_measured(eta=0.85):
    """对比理论公式和实测补偿的远场差异"""

    # G1: 理论准直
    name_theo = "LCP_theoretical_collimate"
    print(f"\n{'=' * 60}")
    print(f"  Test G1: {name_theo} (theoretical formula)")
    print(f"{'=' * 60}")

    init_grid()
    configure_cp(eta=eta, pol="LCP")

    # 使用理论公式
    res = ApplyCollimateLens().call(
        json.dumps({"focal_length": FEED_DISTANCE, "mode": "overwrite"})
    )
    print(f"   {res}")

    # 加载实测幅度 (仅幅度, 不用实测相位)
    import pandas as pd_local

    df = pd_local.read_csv(DEFAULT_LAYOUT_PATH)
    if os.path.exists(LCP_MAG_CSV):
        from agent_tools import parse_measured_csv, interpolate_to_hex_grid

        xm, ym, mag_2d = parse_measured_csv(LCP_MAG_CSV)
        measured_mag = interpolate_to_hex_grid(
            xm, ym, mag_2d, df["x"].values, df["y"].values
        )
        df["measured_mag"] = np.clip(measured_mag, 0, None)
        df["mag"] = df["measured_mag"]
        df.to_csv(DEFAULT_LAYOUT_PATH, index=False)

    pb_and_sim(name_theo)

    # G2: 实测补偿 (已在 Test A 中做过，这里再做一次方便对比)
    test_lcp_collimate(eta=eta)

    print("\n>>> Compare result_LCP_theoretical_collimate_farfield.png")
    print(">>>     vs  result_LCP_collimate_farfield.png")
    print(">>> Measured compensation should show a tighter broadside beam.\n")


# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Metasurface Test Suite — Measured Feed Compensation")
    print(f"  Feed distance: {FEED_DISTANCE * 1000:.0f} mm")
    print(f"  Working freq: {CONSTANTS['freq'] / 1e9:.1f} GHz")
    print(f"  Unit cell: {CONSTANTS['p'] * 1000:.1f} mm")
    print("=" * 60)

    backup_outputs_auto()

    # ===== 核心测试: 实测数据补偿 =====
    test_lcp_collimate(eta=0.85)  # A: LCP → 平面波
    # test_rcp_collimate(eta=0.85)       # B: RCP → 平面波 (需要RCP的CSV文件)

    # ===== 组合功能测试 =====
    # test_lcp_collimate_steer(eta=0.85, theta=20, phi=0)    # C: 补偿+偏转
    # test_lcp_collimate_bessel(eta=0.85, cone=10)            # D: 补偿+Bessel
    # test_lcp_collimate_airy(eta=0.85, coeff=5e6)            # E: 补偿+Airy
    # test_lcp_collimate_dammann(eta=0.85, beam_order=3)      # F: 补偿+达曼

    # ===== 对比测试 =====
    # test_compare_theoretical_vs_measured(eta=0.85)  # G: 理论 vs 实测

    print("\n" + "=" * 60)
    print("  All tests complete. Check outputs/ directory.")
    print("=" * 60)

    # ===== 核心测试: 实测数据补偿 =====
    test_lcp_collimate(eta=0.85)  # A: LCP → 平面波
    # test_rcp_collimate(eta=0.85)       # B: RCP → 平面波 (需要RCP的CSV文件)

    # ===== 组合功能测试 =====
    # test_lcp_collimate_steer(eta=0.85, theta=20, phi=0)    # C: 补偿+偏转
    # test_lcp_collimate_bessel(eta=0.85, cone=10)            # D: 补偿+Bessel
    # test_lcp_collimate_airy(eta=0.85, coeff=5e6)            # E: 补偿+Airy
    # test_lcp_collimate_dammann(eta=0.85, beam_order=3)      # F: 补偿+达曼

    # ===== 对比测试 =====
    # test_compare_theoretical_vs_measured(eta=0.85)  # G: 理论 vs 实测

    print("\n" + "=" * 60)
    print("  All tests complete. Check outputs/ directory.")
    print("=" * 60)
