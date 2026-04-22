import json
import os
from agent_tools import (
    GenerateGridLayout,
    ApplyVortexPhase,
    ApplyDammannGrating,
    CalculatePBRotation,
    SimulateMetasurface,
    init_run_dir,
    get_layout_path,
    RESULTS_ROOT,
)

def run_single_test(beam_order, test_name):
    """
    运行单次测试流程
    beam_order: 3 (for 3x3) or 5 (for 5x5)
    test_name: 用于区分输出目录的标识
    """
    init_run_dir(test_name)

    print(f"\n{'='*20} 开始测试: {test_name} (N={beam_order}) {'='*20}")

    # 1. 初始化阵列 (Hex Grid)
    print(f">>> 1. 初始化阵列...")
    GenerateGridLayout().call(json.dumps({"layout_type": "hex", "radius": 30}))

    # 2. 应用涡旋波相位 (OAM l=1)
    print(f">>> 2. 应用涡旋相位 (l=1)...")
    ApplyVortexPhase().call(json.dumps({"charge_l": 1}))

    # 3. 叠加达曼光栅相位 (Key Step)
    print(f">>> 3. 叠加达曼光栅 (Order={beam_order})...")
    dammann_params = {
        "beam_order": beam_order,
        "period_scale": 12
    }
    res3 = ApplyDammannGrating().call(json.dumps(dammann_params))
    print(res3)

    # 4. 计算 PB 转角
    print(f">>> 4. 计算 PB 转角...")
    CalculatePBRotation().call(json.dumps({}))

    # 5. 仿真与绘图
    print(f">>> 5. 仿真...")
    res5 = SimulateMetasurface().call(json.dumps({}))
    print(res5)

    print(f">>> 测试 {test_name} 完成。结果已保存到 {get_layout_path()} 所在目录")

if __name__ == "__main__":
    # 测试 A: 3x3 波束 (N=3)
    run_single_test(beam_order=3, test_name="3x3_Array")

    # # 测试 B: 5x5 波束 (N=5)
    # run_single_test(beam_order=5, test_name="5x5_Array")

    # (可选) 测试 C: 7x7 波束
    # run_single_test(beam_order=7, test_name="7x7_Array")