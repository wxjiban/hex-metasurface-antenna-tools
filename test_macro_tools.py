import json
import os
import shutil
# 确保这里导入的是你最新的 Tool 文件名
from agent_tools import (
    GenerateGridLayout, 
    ApplyVortexPhase, 
    ApplyDammannGrating, 
    CalculatePBRotation, 
    SimulateMetasurface,
    DEFAULT_LAYOUT_PATH,
    OUTPUT_DIR
)

def run_single_test(beam_order, test_name):
    """
    运行单次测试流程
    beam_order: 3 (for 3x3) or 5 (for 5x5)
    test_name: 用于区分输出文件的后缀
    """
    print(f"\n{'='*20} 开始测试: {test_name} (N={beam_order}) {'='*20}")
    
    # 1. 初始化阵列 (Hex Grid)
    # 每次都重新生成干净的 Grid，防止相位叠加污染
    print(f">>> 1. 初始化阵列...")
    GenerateGridLayout().call(json.dumps({"layout_type": "hex", "radius": 30})) # 半径设大一点以便看清光栅周期

    # 2. 应用涡旋波相位 (OAM l=1)
    print(f">>> 2. 应用涡旋相位 (l=1)...")
    ApplyVortexPhase().call(json.dumps({"charge_l": 1}))

    # 3. 叠加达曼光栅相位 (Key Step)
    print(f">>> 3. 叠加达曼光栅 (Order={beam_order})...")
    # period_scale 设为 30um，确保在阵列上有足够的周期重复
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

    # --- 6. 结果文件重命名备份 ---
    # 因为工具默认都写在 output/current_layout.csv，我们需要把结果拷出来，否则下次循环就覆盖了
    
    # 定义源文件路径 (默认生成的文件)
    src_csv = DEFAULT_LAYOUT_PATH
    # 假设 SimulateMetasurface 生成的图片名是固定的，这里需要根据实际情况调整
    # 通常是 outputs/current_layout_farfield.png 和 outputs/current_layout_structure.png
    src_img_ff = os.path.join(OUTPUT_DIR, "current_layout_farfield.png")
    src_img_st = os.path.join(OUTPUT_DIR, "current_layout_structure.png")

    # 定义目标路径
    dst_csv = os.path.join(OUTPUT_DIR, f"result_{test_name}.csv")
    dst_img_ff = os.path.join(OUTPUT_DIR, f"result_{test_name}_farfield.png")
    dst_img_st = os.path.join(OUTPUT_DIR, f"result_{test_name}_structure.png")

    # 复制并重命名
    if os.path.exists(src_csv): shutil.copy(src_csv, dst_csv)
    if os.path.exists(src_img_ff): shutil.copy(src_img_ff, dst_img_ff)
    if os.path.exists(src_img_st): shutil.copy(src_img_st, dst_img_st)

    print(f">>> 测试 {test_name} 完成。结果已保存为 result_{test_name}_*.png")

if __name__ == "__main__":
    # 测试 A: 3x3 波束 (N=3)
    run_single_test(beam_order=3, test_name="3x3_Array")

    # # 测试 B: 5x5 波束 (N=5)
    # run_single_test(beam_order=5, test_name="5x5_Array")
    
    # (可选) 测试 C: 7x7 波束
    # run_single_test(beam_order=7, test_name="7x7_Array")