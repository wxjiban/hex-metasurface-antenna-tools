import json
import os
import shutil
from agent_tools import (
    GenerateGridLayout,
    ApplyCollimateLens,
    ApplyAxiconPhase,
    ApplyAiryPhase,
    ApplyBeamSteering,
    ApplyVortexPhase,
    ApplyDammannGrating,
    CalculatePBRotation,
    SimulateMetasurface,
    DEFAULT_LAYOUT_PATH,
    OUTPUT_DIR
)


def backup_results(test_name):
    """把默认输出文件拷贝为带名字的备份"""
    pairs = [
        (DEFAULT_LAYOUT_PATH, f"result_{test_name}.csv"),
        (DEFAULT_LAYOUT_PATH.replace(".csv", "_layout.png"), f"result_{test_name}_layout.png"),
        (DEFAULT_LAYOUT_PATH.replace(".csv", "_farfield.png"), f"result_{test_name}_farfield.png"),
    ]
    for src, dst_name in pairs:
        dst = os.path.join(OUTPUT_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy(src, dst)


def init_grid(radius=0.13):
    GenerateGridLayout().call(json.dumps({"layout_type": "hex", "radius": radius}))


def pb_and_sim(test_name):
    CalculatePBRotation().call(json.dumps({}))
    res = SimulateMetasurface().call(json.dumps({}))
    print(res)
    backup_results(test_name)
    print(f">>> [{test_name}] Done.\n")


# ============================================================
# 测试 A: 球面波 → 平面波 (准直)
# ============================================================
def test_collimate(focal_length=0.15):
    name = f"collimate_f{int(focal_length*1000)}mm"
    print(f"\n{'='*50}\n  Test: {name}\n{'='*50}")
    init_grid()
    res = ApplyCollimateLens().call(json.dumps({
        "focal_length": focal_length,
        "mode": "overwrite"
    }))
    print(res)
    pb_and_sim(name)


# ============================================================
# 测试 B: Bessel 非衍射波束
# ============================================================
def test_bessel(cone_angle=10):
    name = f"bessel_{cone_angle}deg"
    print(f"\n{'='*50}\n  Test: {name}\n{'='*50}")
    init_grid()
    res = ApplyAxiconPhase().call(json.dumps({
        "cone_angle_deg": cone_angle,
        "mode": "overwrite"
    }))
    print(res)
    pb_and_sim(name)


# ============================================================
# 测试 C: Airy 自加速波束
# ============================================================
def test_airy(coeff=5e6):
    name = "airy_beam"
    print(f"\n{'='*50}\n  Test: {name}\n{'='*50}")
    init_grid()
    res = ApplyAiryPhase().call(json.dumps({
        "cubic_coeff": coeff,
        "separable": True,
        "mode": "overwrite"
    }))
    print(res)
    pb_and_sim(name)


# ============================================================
# 测试 D: 波束偏转 30°
# ============================================================
def test_steering(theta=30, phi=0):
    name = f"steer_theta{theta}_phi{phi}"
    print(f"\n{'='*50}\n  Test: {name}\n{'='*50}")
    init_grid()
    res = ApplyBeamSteering().call(json.dumps({
        "steer_theta_deg": theta,
        "steer_phi_deg": phi,
        "mode": "overwrite"
    }))
    print(res)
    pb_and_sim(name)


# ============================================================
# 测试 E: 组合功能 — 准直 + 偏转 (球面波入射, 平面波斜出射)
# ============================================================
def test_collimate_then_steer(f=0.15, theta=20, phi=45):
    name = f"collimate_steer_t{theta}_p{phi}"
    print(f"\n{'='*50}\n  Test: {name}\n{'='*50}")
    init_grid()
    
    # 第一步: 准直 (overwrite)
    res1 = ApplyCollimateLens().call(json.dumps({
        "focal_length": f,
        "mode": "overwrite"
    }))
    print(res1)
    
    # 第二步: 偏转 (add, 叠加到准直相位上)
    res2 = ApplyBeamSteering().call(json.dumps({
        "steer_theta_deg": theta,
        "steer_phi_deg": phi,
        "mode": "add"
    }))
    print(res2)
    
    pb_and_sim(name)


# ============================================================
# 测试 F: 组合功能 — 准直 + Bessel (球面波入射, 非衍射输出)
# ============================================================
def test_collimate_then_bessel(f=0.15, cone=8):
    name = f"collimate_bessel_{cone}deg"
    print(f"\n{'='*50}\n  Test: {name}\n{'='*50}")
    init_grid()
    
    res1 = ApplyCollimateLens().call(json.dumps({
        "focal_length": f,
        "mode": "overwrite"
    }))
    print(res1)
    
    res2 = ApplyAxiconPhase().call(json.dumps({
        "cone_angle_deg": cone,
        "mode": "add"
    }))
    print(res2)
    
    pb_and_sim(name)


# ============================================================

if __name__ == "__main__":
    # 基础功能测试
    test_collimate(focal_length=0.15)       # A: 准直
    test_bessel(cone_angle=10)              # B: Bessel
    test_airy(coeff=5e6)                    # C: Airy
    test_steering(theta=30, phi=0)          # D: 偏转

    # 组合功能测试 (创新点)
    test_collimate_then_steer(f=0.15, theta=20, phi=45)   # E: 准直+偏转
    test_collimate_then_bessel(f=0.15, cone=8)             # F: 准直+Bessel