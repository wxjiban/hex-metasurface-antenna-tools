import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import traceback
import json
from datetime import datetime
# 尝试导入qwen_agent，如果不可用则提供桩
try:
    from qwen_agent.tools.base import BaseTool, register_tool
except ImportError:
    # 独立运行时的兼容桩
    def register_tool(name):
        def decorator(cls):
            cls.tool_name = name
            return cls

        return decorator

    class BaseTool:
        def call(self, params, **kwargs):
            raise NotImplementedError


# ================= 全局配置 =================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_ROOT, exist_ok=True)

_run_dir = None
_run_prefix = None


def init_run_dir(test_name: str) -> str:
    """创建本次运行的结果目录，所有输出直接写入这里"""
    global _run_dir, _run_prefix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _run_prefix = f"{timestamp}_{test_name}"
    _run_dir = os.path.join(RESULTS_ROOT, _run_prefix)
    os.makedirs(_run_dir, exist_ok=True)
    print(f"[RUN DIR] {_run_dir}")
    return _run_dir


def get_layout_path() -> str:
    """返回当前运行的 layout CSV 路径"""
    if _run_dir is None or _run_prefix is None:
        raise RuntimeError("请先调用 init_run_dir()")
    return os.path.join(_run_dir, f"{_run_prefix}.csv")

CONSTANTS = {
    "freq": 9e9,
    "c": 3e8,
    "p": 7.2e-3,  # 单元周期 7.2mm
    "lambda": 3e8 / 9e9,
    "k": 2 * math.pi / (3e8 / 9e9),
}


def _phase_normalize(phase):
    return phase % (2 * np.pi)


# ================= 辅助函数: 解析实测 CSV =================


def parse_measured_csv(csv_path):
    """
    解析仿真导出的 23x23 幅度/相位 CSV 文件。
    第一行为列坐标(带mm), 第一列为行坐标(带mm), 数据区为幅度或相位值。
    返回: x_coords(m), y_coords(m), data_2d(23x23 ndarray)
    """
    df_raw = pd.read_csv(csv_path, index_col=0)

    # 解析列坐标 (去掉 "mm" 后缀, 转为米)
    x_coords = np.array([float(c.replace("mm", "")) * 1e-3 for c in df_raw.columns])
    # 解析行坐标
    y_coords = np.array([float(str(r).replace("mm", "")) * 1e-3 for r in df_raw.index])
    # 数据矩阵
    data_2d = df_raw.values.astype(float)

    return x_coords, y_coords, data_2d


def interpolate_to_hex_grid(x_coords, y_coords, data_2d, hex_x, hex_y):
    """
    将规则网格上的数据插值到六边形阵列坐标上。
    使用 scipy RegularGridInterpolator, 边界外用最近邻外推。
    """
    from scipy.interpolate import RegularGridInterpolator

    # 注意: RegularGridInterpolator 要求坐标严格递增
    # data_2d[i,j] 对应 y_coords[i], x_coords[j]
    # 确保坐标递增
    if y_coords[0] > y_coords[-1]:
        y_coords = y_coords[::-1]
        data_2d = data_2d[::-1, :]
    if x_coords[0] > x_coords[-1]:
        x_coords = x_coords[::-1]
        data_2d = data_2d[:, ::-1]

    interp = RegularGridInterpolator(
        (y_coords, x_coords),
        data_2d,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    # 查询点 (y, x) 顺序
    pts = np.column_stack([hex_y, hex_x])
    return interp(pts)


# ================= Tool 0: 阵列初始化 =================


@register_tool("generate_grid_layout")
class GenerateGridLayout(BaseTool):
    description = (
        "Initialize the metasurface array layout. "
        "Call this FIRST to create the coordinate system."
    )
    parameters = [
        {
            "name": "layout_type",
            "type": "string",
            "description": 'Type of grid: "hex" or "rect". Default "hex".',
            "required": False,
        },
        {
            "name": "radius",
            "type": "number",
            "description": "Aperture radius in meters. Default 0.13.",
            # 两种尺寸 0.13 0.08
            "required": False,
        },
        {
            "name": "output_path",
            "type": "string",
            "description": "Optional save path. Default: current run dir/layout.csv",
            "required": False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json

            p = json.loads(params)
            layout_type = p.get("layout_type", "hex")
            radius = float(p.get("radius", 0.13))
            save_path = p.get("output_path") or get_layout_path()
            period = CONSTANTS["p"]
            xyset = set()

            if layout_type == "hex":
                x, y = period / 2, 0
                xyset.add((x, y))
                steps = math.ceil(radius / period)
                for _ in range(steps):
                    new_points = set()
                    for cx, cy in xyset:
                        for ang in [0, 60, 120, 180, 240, 300]:
                            rad = math.radians(ang)
                            nx = round(cx + period * math.cos(rad), 10)
                            ny = round(cy + period * math.sin(rad), 10)
                            if nx**2 + ny**2 < radius**2:
                                new_points.add((nx, ny))
                    xyset |= new_points

            df = pd.DataFrame(list(xyset), columns=["x", "y"])
            df["target_phase"] = 0.0
            df["rotation_angle_deg"] = 0.0
            df.to_csv(save_path, index=False)
            return f"Layout initialized ({layout_type}). Total Units: {len(df)}. Saved to: {save_path}"
        except Exception as e:
            return f"Error: {str(e)}"


# ================= Tool 1: 涡旋波 =================


@register_tool("apply_vortex_phase")
class ApplyVortexPhase(BaseTool):
    description = "Apply Vortex Phase (OAM) to the layout."
    parameters = [
        {
            "name": "charge_l",
            "type": "integer",
            "description": "Topological charge.",
            "required": True,
        },
        {
            "name": "file_path",
            "type": "string",
            "description": "Optional path to CSV.",
            "required": False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json

            p = json.loads(params)
            l = int(p.get("charge_l", 1))
            file_path = p.get("file_path") or get_layout_path()
            if not os.path.exists(file_path):
                return f"Error: File {file_path} not found."
            df = pd.read_csv(file_path)
            theta = np.arctan2(df["y"], df["x"] + 1e-9)
            vortex_phase = l * theta
            if "target_phase" not in df.columns:
                df["target_phase"] = 0.0
            df["target_phase"] += vortex_phase
            df["target_phase"] = df["target_phase"] % (2 * np.pi)
            df.to_csv(file_path, index=False)
            return f"Vortex Phase (l={l}) applied to {len(df)} units."
        except Exception as e:
            return f"Error: {str(e)}"


# ================= Tool 2: 达曼光栅 =================


@register_tool("apply_dammann_grating")
class ApplyDammannGrating(BaseTool):
    description = "Apply Dammann Grating phase for NxN beam splitting."
    parameters = [
        {
            "name": "file_path",
            "type": "string",
            "description": "Optional CSV path.",
            "required": False,
        },
        {
            "name": "beam_order",
            "type": "integer",
            "description": "Beam order N. Supports [2-7].",
            "required": True,
        },
        {
            "name": "period_multiple",
            "type": "integer",
            "description": "Grating period in unit cells. Default 12.",
            "required": False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json

            p_dict = json.loads(params)
            file_path = p_dict.get("file_path") or get_layout_path()
            N = int(p_dict.get("beam_order", 5))
            M = int(p_dict.get("period_multiple", 12))
            unit_p = CONSTANTS["p"]
            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}."
            df = pd.read_csv(file_path)
            sqrt3 = np.sqrt(3)
            r_float = 2 * df["y"].values / (sqrt3 * unit_p)
            r = np.round(r_float).astype(int)
            q_float = df["x"].values / unit_p - r_float / 2
            q = np.round(q_float).astype(int)
            u_idx, v_idx = q, r

            raw_transitions = {
                2: [0.5],
                3: [0.2656],
                4: [0.1767, 0.4577],
                5: [0.1322, 0.4808],
                6: [0.1084, 0.2941, 0.4519],
                7: [0.0886, 0.3060, 0.4495],
            }
            if N not in raw_transitions:
                return f"Error: Beam order {N} not supported."
            raw_points = raw_transitions[N]
            if N == 2:
                full_points = [0.5]
            else:
                full_points = sorted(
                    list(set(raw_points + [1.0 - pt for pt in raw_points]))
                )

            def get_phase_from_index(index_array, period_int, pts):
                norm = (index_array % period_int) / period_int
                flip_counts = np.zeros_like(norm, dtype=int)
                for pt in pts:
                    flip_counts += (norm > pt).astype(int)
                return np.where(flip_counts % 2 != 0, np.pi, 0.0)

            phi_u = get_phase_from_index(u_idx, M, full_points)
            phi_v = get_phase_from_index(v_idx, M, full_points)
            total_dammann_phase = phi_u + phi_v

            if "target_phase" not in df.columns:
                df["target_phase"] = 0.0
            df["target_phase"] += total_dammann_phase
            df["target_phase"] = df["target_phase"] % (2 * np.pi)
            df.to_csv(file_path, index=False)
            return f"Applied {N}x{N} Dammann Grating. Period M={M}. Points={np.round(full_points, 4)}"
        except Exception as e:
            import traceback

            return f"Error: {traceback.format_exc()}"


# ================= Tool 5: 球面波准直 (理论公式版, 保留) =================


@register_tool("apply_collimate_lens")
class ApplyCollimateLens(BaseTool):
    description = (
        "Apply collimating lens phase (theoretical). phi = k*(sqrt(x^2+y^2+f^2) - f)."
    )
    parameters = [
        {
            "name": "focal_length",
            "type": "number",
            "description": "Focal distance (m). Default 0.15.",
            "required": False,
        },
        {
            "name": "mode",
            "type": "string",
            "description": '"overwrite" or "add". Default "overwrite".',
            "required": False,
        },
        {
            "name": "file_path",
            "type": "string",
            "description": "Optional CSV path.",
            "required": False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json

            p = json.loads(params)
            f = float(p.get("focal_length", 0.15))
            mode = p.get("mode", "overwrite")
            file_path = p.get("file_path") or get_layout_path()
            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}."
            df = pd.read_csv(file_path)
            k = CONSTANTS["k"]
            x, y = df["x"].values, df["y"].values
            collimate_phase = k * (np.sqrt(x**2 + y**2 + f**2) - f)
            if "target_phase" not in df.columns:
                df["target_phase"] = 0.0
            if mode == "overwrite":
                df["target_phase"] = collimate_phase
            else:
                df["target_phase"] += collimate_phase
            df["target_phase"] = df["target_phase"] % (2 * np.pi)
            df.to_csv(file_path, index=False)
            return f"Collimating Lens applied (f={f:.4f}m). Units: {len(df)}"
        except Exception as e:
            import traceback

            return f"Error: {traceback.format_exc()}"


# ================= Tool 6: Bessel波束 =================


@register_tool("apply_axicon_phase")
class ApplyAxiconPhase(BaseTool):
    description = "Generate Bessel beam via axicon phase."
    parameters = [
        {
            "name": "cone_angle_deg",
            "type": "number",
            "description": "Half-angle (deg). Default 10.",
            "required": False,
        },
        {
            "name": "xy_unit",
            "type": "string",
            "description": "'m' or 'mm'. Must match the unit of x/y in the CSV. Default 'm'.",
            "required": False,
        },
        {
            "name": "mode",
            "type": "string",
            "description": '"overwrite" or "add". Default "overwrite".',
            "required": False,
        },
        {
            "name": "file_path",
            "type": "string",
            "description": "Optional CSV path.",
            "required": False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            p = json.loads(params)
            alpha_deg = float(p.get("cone_angle_deg", 10.0))
            xy_unit   = p.get("xy_unit", "m")
            mode      = p.get("mode", "overwrite")
            file_path = p.get("file_path") or get_layout_path()

            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}."

            df = pd.read_csv(file_path)
            x_raw = df["x"].values
            y_raw = df["y"].values

            # ✅ 统一转换到 m，与 k (rad/m) 保持一致
            if xy_unit == "mm":
                x_m = x_raw * 1e-3
                y_m = y_raw * 1e-3
            elif xy_unit == "m":
                x_m = x_raw
                y_m = y_raw
            else:
                return f"Error: xy_unit must be 'm' or 'mm', got '{xy_unit}'."

            # ✅ k 单位 rad/m，与转换后的坐标（m）一致
            k     = CONSTANTS["k"]          # rad/m
            lam   = CONSTANTS["lambda"]     # m
            alpha = np.radians(alpha_deg)
            rho   = np.sqrt(x_m**2 + y_m**2)   # m

            # ✅ Axicon相位公式
            axicon_phase = -k * np.sin(alpha) * rho   # rad

            # ✅ 诊断信息
            R      = rho.max()
            z_max  = R / np.tan(alpha)               # Bessel非衍射距离 (m)
            z_max_approx = R / np.sin(alpha)         # 近似值
            phase_cycles = (k * np.sin(alpha) * R) / (2 * np.pi)

            print(f"[Axicon Diagnostics]")
            print(f"  λ          = {lam*1e3:.2f} mm")
            print(f"  k          = {k:.2f} rad/m")
            print(f"  α          = {alpha_deg}°")
            print(f"  R (半径)    = {R*1e3:.1f} mm")
            print(f"  相位周期数  = {phase_cycles:.1f} cycles")
            print(f"  非衍射距离  = {z_max*1e3:.1f} mm  (tan近似)")
            print(f"             ≈ {z_max_approx*1e3:.1f} mm  (sin近似)")

            if phase_cycles < 2:
                print(f"[WARNING] 相位周期数 < 2，Bessel效果弱，建议增大 cone_angle_deg")
            elif phase_cycles > 200:
                print(f"[WARNING] 相位周期数 > 200，采样可能不足")
            else:
                print(f"[OK] 相位周期数合理")

            # ✅ wrapping
            wrapped = axicon_phase % (2 * np.pi)

            if "target_phase" not in df.columns:
                df["target_phase"] = 0.0

            if mode == "overwrite":
                df["target_phase"] = wrapped
            else:
                df["target_phase"] = (df["target_phase"] + axicon_phase) % (2 * np.pi)

            df.to_csv(file_path, index=False)

            return (
                f"Axicon phase applied. "
                f"cone={alpha_deg}°, "
                f"xy_unit='{xy_unit}', "
                f"λ={lam*1e3:.2f}mm, "
                f"phase_cycles={phase_cycles:.1f}, "
                f"z_max={z_max*1e3:.1f}mm, "
                f"units={len(df)}"
            )

        except Exception as e:
            return f"Error: {traceback.format_exc()}"
# ================= Tool 7: Airy波束 =================


@register_tool("apply_airy_phase")
class ApplyAiryPhase(BaseTool):
    description = "Generate self-accelerating Airy beam via cubic phase."
    parameters = [
        {
            "name": "cubic_coeff",
            "type": "number",
            "description": (
                "Cubic coefficient in rad/m³ (if xy_unit='m') or rad/mm³ (if xy_unit='mm'). "
                "Recommended ~8.6e4 for m, ~8.6e-5 for mm (10 phase cycles over 90mm aperture)."
            ),
            "required": False,
        },
        {
            "name": "xy_unit",
            "type": "string",
            "description": "'m' or 'mm'. Must match the unit of x/y in the CSV. Default 'm'.",
            "required": False,
        },
        {
            "name": "n_cycles",
            "type": "number",
            "description": "Target phase cycles across half-aperture. Used only if cubic_coeff is not provided. Default 10.",
            "required": False,
        },
        {
            "name": "separable",
            "type": "boolean",
            "description": "True: phi=a3*(x³+y³); False: only x direction phi=a3*x³. Default True.",
            "required": False,
        },
        {
            "name": "mode",
            "type": "string",
            "description": '"overwrite" or "add". Default "overwrite".',
            "required": False,
        },
        {
            "name": "file_path",
            "type": "string",
            "description": "Optional CSV path.",
            "required": False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            p = json.loads(params)
            xy_unit  = p.get("xy_unit", "m")
            separable = p.get("separable", True)
            mode     = p.get("mode", "overwrite")
            file_path = p.get("file_path") or get_layout_path()

            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}."

            df = pd.read_csv(file_path)
            x_raw = df["x"].values
            y_raw = df["y"].values

            # ✅ 统一转换到 m 进行计算
            if xy_unit == "mm":
                x_m = x_raw * 1e-3
                y_m = y_raw * 1e-3
            elif xy_unit == "m":
                x_m = x_raw
                y_m = y_raw
            else:
                return f"Error: xy_unit must be 'm' or 'mm', got '{xy_unit}'."

            # ✅ 自动推算 coeff（若未提供）
            if "cubic_coeff" in p:
                a3 = float(p["cubic_coeff"])
            else:
                n_cycles = float(p.get("n_cycles", 10))
                x_max = np.max(np.abs(x_m))
                if x_max < 1e-9:
                    return "Error: x values are all zero."
                a3 = n_cycles * 2 * np.pi / (x_max ** 3)
                print(f"[AutoCoeff] x_max={x_max:.4f}m, "
                      f"n_cycles={n_cycles}, a3={a3:.3e} rad/m³")

            # ✅ 计算Airy相位（单位：rad）
            if separable:
                airy_phase = a3 * (x_m**3 + y_m**3)   # 二维可分
            else:
                airy_phase = a3 * (x_m**3)              # 仅x方向自加速

            # ✅ 诊断信息
            phase_range = airy_phase.max() - airy_phase.min()
            n_cycles_actual = phase_range / (2 * np.pi)

            if n_cycles_actual > 100:
                print(f"[WARNING] Phase cycles={n_cycles_actual:.1f} > 100, "
                      f"may be under-sampled! Consider reducing cubic_coeff.")
            elif n_cycles_actual < 2:
                print(f"[WARNING] Phase cycles={n_cycles_actual:.1f} < 2, "
                      f"modulation too weak! Consider increasing cubic_coeff.")
            else:
                print(f"[OK] Phase cycles={n_cycles_actual:.1f}, looks reasonable.")

            # ✅ wrapping 到 [0, 2π]
            wrapped = airy_phase % (2 * np.pi)

            if "target_phase" not in df.columns:
                df["target_phase"] = 0.0

            if mode == "overwrite":
                df["target_phase"] = wrapped
            else:
                df["target_phase"] = (df["target_phase"] + airy_phase) % (2 * np.pi)

            df.to_csv(file_path, index=False)

            return (
                f"Airy phase applied. "
                f"a3={a3:.4e} rad/m³, "
                f"xy_unit='{xy_unit}', "
                f"separable={separable}, "
                f"phase_range={phase_range:.2f} rad, "
                f"cycles={n_cycles_actual:.1f}, "
                f"units={len(df)}"
            )

        except Exception as e:
            return f"Error: {traceback.format_exc()}"
# ================= Tool 8: 波束偏转 =================


@register_tool("apply_beam_steering")
class ApplyBeamSteering(BaseTool):
    description = "Steer beam to (theta, phi) via linear phase gradient."
    parameters = [
        {
            "name": "steer_theta_deg",
            "type": "number",
            "description": "Elevation angle (deg). Default 30.",
            "required": False,
        },
        {
            "name": "steer_phi_deg",
            "type": "number",
            "description": "Azimuth angle (deg). Default 0.",
            "required": False,
        },
        {
            "name": "mode",
            "type": "string",
            "description": '"overwrite" or "add".',
            "required": False,
        },
        {
            "name": "file_path",
            "type": "string",
            "description": "Optional CSV path.",
            "required": False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json

            p = json.loads(params)
            theta_d = float(p.get("steer_theta_deg", 30.0))
            phi_d = float(p.get("steer_phi_deg", 0.0))
            mode = p.get("mode", "overwrite")
            file_path = p.get("file_path") or get_layout_path()
            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}."
            df = pd.read_csv(file_path)
            k = CONSTANTS["k"]
            theta, phi = np.radians(theta_d), np.radians(phi_d)
            x, y = df["x"].values, df["y"].values
            steer_phase = k * (
                x * np.sin(theta) * np.cos(phi) + y * np.sin(theta) * np.sin(phi)
            )
            if "target_phase" not in df.columns:
                df["target_phase"] = 0.0
            if mode == "overwrite":
                df["target_phase"] = steer_phase
            else:
                df["target_phase"] += steer_phase
            df["target_phase"] = df["target_phase"] % (2 * np.pi)
            df.to_csv(file_path, index=False)
            return f"Beam steering applied. θ={theta_d}°, φ={phi_d}°. Units: {len(df)}"
        except Exception as e:
            import traceback

            return f"Error: {traceback.format_exc()}"


# ================= Tool 9: CP效率配置 =================


@register_tool("configure_cp_efficiency")
class ConfigureCPEfficiency(BaseTool):
    description = (
        "Configure circular polarization conversion efficiency and handedness."
    )
    parameters = [
        {
            "name": "eta",
            "type": "number",
            "description": "Conversion efficiency 0~1. Default 0.85.",
            "required": False,
        },
        {
            "name": "incident_pol",
            "type": "string",
            "description": '"LCP" or "RCP". Default "LCP".',
            "required": False,
        },
        {
            "name": "file_path",
            "type": "string",
            "description": "Optional CSV path.",
            "required": False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json

            p = json.loads(params)
            eta = float(p.get("eta", 0.85))
            pol = p.get("incident_pol", "LCP").upper()
            file_path = p.get("file_path") or get_layout_path()
            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}."
            df = pd.read_csv(file_path)
            df["cp_eta"] = eta
            df["cp_sign"] = 1.0 if pol == "LCP" else -1.0
            df.to_csv(file_path, index=False)
            return f"CP config: {pol}, η={eta:.2f}"
        except Exception as e:
            import traceback

            return f"Error: {traceback.format_exc()}"


# =================================================================
# Tool 10: 基于实测数据的球面波补偿 (核心新功能)
# =================================================================


@register_tool("apply_measured_compensation")
class ApplyMeasuredCompensation(BaseTool):
    description = (
        "Load measured horn feed illumination data (magnitude + phase CSV files) "
        "and compute the compensation phase for each hex unit cell. "
        "The compensation phase = 2π - measured_phase (mod 2π), so that the "
        "total transmitted wavefront becomes a uniform plane wave. "
        "Also stores the measured magnitude per unit for accurate far-field simulation."
    )

    parameters = [
        {
            "name": "phase_csv",
            "type": "string",
            "description": "Path to the measured phase CSV file (23x23, values in radians, coords in mm).",
            "required": True,
        },
        {
            "name": "magnitude_csv",
            "type": "string",
            "description": "Path to the measured magnitude CSV file (same format). "
            "If not provided, assumes uniform illumination.",
            "required": False,
        },
        {
            "name": "additional_phase_csv",
            "type": "string",
            "description": "Optional: path to a second phase profile to ADD on top of compensation "
            "(e.g., for beam steering or vortex after collimation).",
            "required": False,
        },
        {
            "name": "file_path",
            "type": "string",
            "description": "Path to the hex layout CSV. Default: current layout.",
            "required": False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json

            p = json.loads(params)
            phase_csv = p["phase_csv"]
            mag_csv = p.get("magnitude_csv")
            extra_csv = p.get("additional_phase_csv")
            file_path = p.get("file_path") or get_layout_path()

            if not os.path.exists(file_path):
                return f"Error: Layout file not found: {file_path}"
            if not os.path.exists(phase_csv):
                return f"Error: Phase CSV not found: {phase_csv}"

            df = pd.read_csv(file_path)
            hex_x = df["x"].values
            hex_y = df["y"].values

            # --- 解析实测相位 ---
            x_coords, y_coords, phase_2d = parse_measured_csv(phase_csv)
            measured_phase = interpolate_to_hex_grid(
                x_coords, y_coords, phase_2d, hex_x, hex_y
            )

            # --- 解析实测幅度 ---
            if mag_csv and os.path.exists(mag_csv):
                xm, ym, mag_2d = parse_measured_csv(mag_csv)
                measured_mag = interpolate_to_hex_grid(xm, ym, mag_2d, hex_x, hex_y)
                measured_mag = np.clip(measured_mag, 0, None)  # 幅度不能为负
            else:
                measured_mag = np.ones(len(df))

            # --- 计算补偿相位 ---
            # 目标: 出射后等相面 → 需要补偿掉入射相位的不均匀性
            # 补偿相位 = -measured_phase (mod 2π) = (2π - measured_phase) mod 2π
            compensation_phase = (2 * np.pi - measured_phase) % (2 * np.pi)

            # --- 存储 ---
            df["measured_phase"] = measured_phase % (2 * np.pi)
            df["measured_mag"] = measured_mag
            df["compensation_phase"] = compensation_phase
            df["target_phase"] = compensation_phase  # 初始 target = 纯补偿

            # --- 可选: 叠加额外相位 ---
            if extra_csv and os.path.exists(extra_csv):
                xe, ye, extra_2d = parse_measured_csv(extra_csv)
                extra_phase = interpolate_to_hex_grid(xe, ye, extra_2d, hex_x, hex_y)
                df["target_phase"] = (df["target_phase"] + extra_phase) % (2 * np.pi)

            df["target_phase"] = df["target_phase"] % (2 * np.pi)
            df.to_csv(file_path, index=False)

            return (
                f"Measured compensation applied.\n"
                f" - Phase CSV: {phase_csv}\n"
                f" - Magnitude CSV: {mag_csv or 'uniform'}\n"
                f" - Units interpolated: {len(df)}\n"
                f" - Measured phase range: [{np.min(measured_phase):.3f}, {np.max(measured_phase):.3f}] rad\n"
                f" - Measured mag range: [{np.min(measured_mag):.4f}, {np.max(measured_mag):.4f}]\n"
                f" - Compensation phase set as target_phase."
            )
        except Exception as e:
            import traceback

            return f"Error: {traceback.format_exc()}"


# ================= Tool 3 (修正版): PB 相位 =================


@register_tool("calculate_pb_rotation")
class CalculatePBRotation(BaseTool):
    description = (
        "PB Phase: convert target_phase -> rotation angle. "
        "LCP: φ=+2θ → θ=φ/2. RCP: φ=-2θ → θ=-φ/2. "
        "Reads cp_sign from layout (set by ConfigureCPEfficiency)."
    )
    parameters = [
        {
            "name": "file_path",
            "type": "string",
            "description": "Optional CSV path.",
            "required": False,
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json

            p = json.loads(params)
            file_path = p.get("file_path") or get_layout_path()
            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}."
            df = pd.read_csv(file_path)
            if "target_phase" not in df.columns:
                return "Error: No 'target_phase' found."

            cp_sign = df["cp_sign"].iloc[0] if "cp_sign" in df.columns else 1.0

            df["rotation_angle_rad"] = df["target_phase"] / (2.0 * cp_sign)
            df["rotation_angle_deg"] = np.degrees(df["rotation_angle_rad"])

            # 幅度: 优先使用实测幅度
            if "measured_mag" in df.columns:
                df["mag"] = df["measured_mag"]
            else:
                intercept, slope = 0.000954, 0.9225
                df["mag"] = df["target_phase"] * intercept + slope

            df.to_csv(file_path, index=False)
            pol_str = "LCP→RCP" if cp_sign > 0 else "RCP→LCP"
            return f"PB Rotation calculated ({pol_str}). Updated {file_path}"
        except Exception as e:
            return f"Error: {str(e)}"


# ================= Tool 4 (修正版): CP 远场仿真 =================


@register_tool("simulate_metasurface")
class SimulateMetasurface(BaseTool):
    description = (
        "CP-aware far-field simulation. Models cross-pol (designed beam) and "
        "co-pol leakage. Produces structure map, 3D far-field, UV pattern. "
        "Uses measured magnitude if available."
    )
    parameters = [
        {
            "name": "file_path",
            "type": "string",
            "description": "Optional CSV path.",
            "required": False,
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json

            p = json.loads(params)
            file_path = p.get("file_path") or get_layout_path()
            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}."

            df = pd.read_csv(file_path)
            xy = df[["x", "y"]].values
            phase_data = (
                df["target_phase"].values
                if "target_phase" in df.columns
                else np.zeros(len(df))
            )
            mags = df["mag"].values if "mag" in df.columns else np.ones(len(df))

            eta = df["cp_eta"].iloc[0] if "cp_eta" in df.columns else 0.85
            cp_sign = df["cp_sign"].iloc[0] if "cp_sign" in df.columns else 1.0

            amp_cross = np.sqrt(eta)
            amp_co = np.sqrt(1.0 - eta)

            pol_label = "LCP→RCP" if cp_sign > 0 else "RCP→LCP"

            # ===== 1. 结构图: 相位分布 + 幅度分布 =====
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            radius_hex = CONSTANTS["p"] / math.sqrt(3)

            for ax_idx, (data_arr, title, cmap_name, vmin, vmax) in enumerate(
                [
                    (
                        phase_data,
                        f"Compensation Phase ({pol_label})",
                        "jet",
                        0,
                        2 * np.pi,
                    ),
                    (mags, "Illumination Amplitude", "hot", 0, np.max(mags) + 0.01),
                ]
            ):
                ax = axes[ax_idx]
                ax.set_aspect("equal")
                norm_c = colors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.get_cmap(cmap_name)
                patches = []
                colors_list = []
                for idx, (x, y) in enumerate(xy):
                    c = cmap(norm_c(data_arr[idx]))
                    poly = RegularPolygon(
                        (x, y), numVertices=6, radius=radius_hex, orientation=0
                    )
                    patches.append(poly)
                    colors_list.append(c)
                pc = PatchCollection(
                    patches, facecolors=colors_list, edgecolors="grey", linewidths=0.1
                )
                ax.add_collection(pc)
                ax.autoscale_view()
                ax.axis("off")
                ax.set_title(title, fontsize=12)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
                plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

            plt.suptitle(f"Metasurface Layout (η={eta:.2f})", fontsize=14)
            plt.tight_layout()
            img_struct = file_path.replace(".csv", "_layout.png")
            plt.savefig(img_struct, dpi=300, bbox_inches="tight")
            plt.close()

            # ===== 2. 入射相位分布图 (如果有实测数据) =====
            img_incident = None
            if "measured_phase" in df.columns:
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                for ax_idx, (col, title, cmap_name) in enumerate(
                    [
                        ("measured_phase", "Measured Incident Phase", "hsv"),
                        ("compensation_phase", "Compensation Phase", "hsv"),
                    ]
                ):
                    if col not in df.columns:
                        continue
                    ax = axes[ax_idx]
                    ax.set_aspect("equal")
                    d = df[col].values
                    norm_c = colors.Normalize(vmin=0, vmax=2 * np.pi)
                    cmap = plt.get_cmap(cmap_name)
                    patches = []
                    colors_list = []
                    for idx, (x, y) in enumerate(xy):
                        c = cmap(norm_c(d[idx]))
                        poly = RegularPolygon(
                            (x, y), numVertices=6, radius=radius_hex, orientation=0
                        )
                        patches.append(poly)
                        colors_list.append(c)
                    pc = PatchCollection(
                        patches,
                        facecolors=colors_list,
                        edgecolors="grey",
                        linewidths=0.1,
                    )
                    ax.add_collection(pc)
                    ax.autoscale_view()
                    ax.axis("off")
                    ax.set_title(title, fontsize=12)
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
                    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

                plt.suptitle("Incident vs Compensation Phase", fontsize=14)
                plt.tight_layout()
                img_incident = file_path.replace(".csv", "_incident_vs_comp.png")
                plt.savefig(img_incident, dpi=300, bbox_inches="tight")
                plt.close()

            # ===== 3. 远场仿真 =====
            k = CONSTANTS["k"]
            M_grid, N_grid = 120, 120
            theta_arr = np.linspace(0, np.radians(60), M_grid)
            phi_arr = np.linspace(0, 2 * np.pi, N_grid)
            THETA, PHI = np.meshgrid(theta_arr, phi_arr)
            u = np.sin(THETA) * np.cos(PHI)
            v = np.sin(THETA) * np.sin(PHI)

            X, Y = xy[:, 0], xy[:, 1]
            E_cross = np.zeros_like(THETA, dtype=complex)
            E_co = np.zeros_like(THETA, dtype=complex)

            batch_size = 200
            for i in range(0, len(X), batch_size):
                end = min(i + batch_size, len(X))
                x_b = X[i:end, None, None]
                y_b = Y[i:end, None, None]
                p_b = phase_data[i:end, None, None]
                m_b = mags[i:end, None, None]
                path_diff = k * (x_b * u + y_b * v)

                E_cross += np.sum(
                    m_b * amp_cross * np.exp(1j * (p_b + path_diff)), axis=0
                )
                E_co += np.sum(m_b * amp_co * np.exp(1j * path_diff), axis=0)

            global_max = max(np.max(np.abs(E_cross)), np.max(np.abs(E_co)), 1e-9)
            E_cross_norm = np.abs(E_cross) / global_max
            E_co_norm = np.abs(E_co) / global_max
            E_total_norm = np.sqrt(E_cross_norm**2 + E_co_norm**2)
            E_total_norm /= np.max(E_total_norm) + 1e-9

            # --- 绘图 2x2 ---
            fig = plt.figure(figsize=(18, 16))

            def _plot_3d(ax, E_data, title, cmap_name="jet"):
                Xp = E_data * np.sin(THETA) * np.cos(PHI)
                Yp = E_data * np.sin(THETA) * np.sin(PHI)
                Zp = E_data * np.cos(THETA)
                surf = ax.plot_surface(
                    Xp, Yp, Zp, cmap=cmap_name, rcount=80, ccount=80, alpha=0.9
                )
                ax.set_title(title, fontsize=11, pad=15)
                return surf

            ax1 = fig.add_subplot(221, projection="3d")
            s1 = _plot_3d(ax1, E_cross_norm, f"Cross-pol ({pol_label})\nη={eta:.2f}")
            fig.colorbar(s1, ax=ax1, shrink=0.4)

            ax2 = fig.add_subplot(222, projection="3d")
            s2 = _plot_3d(ax2, E_co_norm, f"Co-pol Leakage\n(1-η)={1 - eta:.2f}", "hot")
            fig.colorbar(s2, ax=ax2, shrink=0.4)

            ax3 = fig.add_subplot(223, projection="3d")
            s3 = _plot_3d(ax3, E_total_norm, "Total Field")
            fig.colorbar(s3, ax=ax3, shrink=0.4)

            ax4 = fig.add_subplot(224)
            u_2d = np.sin(THETA) * np.cos(PHI)
            v_2d = np.sin(THETA) * np.sin(PHI)
            E_cross_dB = 20 * np.log10(E_cross_norm + 1e-9)
            E_cross_dB = np.clip(E_cross_dB, -30, 0)
            pcm = ax4.pcolormesh(u_2d, v_2d, E_cross_dB, cmap="jet", shading="gouraud")
            ax4.set_xlabel("u = sinθ·cosφ")
            ax4.set_ylabel("v = sinθ·sinφ")
            ax4.set_title("Cross-pol UV Pattern (dB)", fontsize=11)
            ax4.set_aspect("equal")
            ax4.set_xlim(-1, 1)
            ax4.set_ylim(-1, 1)
            fig.colorbar(pcm, ax=ax4, label="dB")

            plt.suptitle(f"CP Far-field ({pol_label}, η={eta:.2f})", fontsize=14)
            plt.tight_layout()
            img_farfield = file_path.replace(".csv", "_farfield.png")
            plt.savefig(img_farfield, dpi=150, bbox_inches="tight")
            plt.close()

            results = [
                f"CP Simulation Done ({pol_label}, η={eta:.2f}).",
                f" - Structure: {img_struct}",
                f" - Far-field: {img_farfield}",
            ]
            if img_incident:
                results.append(f" - Incident/Comp: {img_incident}")

            return "\n".join(results)
        except Exception as e:
            import traceback

            return f"Sim Error: {traceback.format_exc()}"


# ================= Tool 11: 近场传播仿真 (角谱法) =================


@register_tool("simulate_nearfield_propagation")
class SimulateNearfieldPropagation(BaseTool):
    description = "Near-field propagation via Angular Spectrum Method. Visualizes XOY slices and XOZ cross-section for Bessel/Airy beam verification."
    parameters = [
        {
            "name": "z_max",
            "type": "number",
            "description": "Max propagation distance in meters. Default 0.5.",
            "required": False,
        },
        {
            "name": "z_steps",
            "type": "integer",
            "description": "Number of z slices. Default 50.",
            "required": False,
        },
        {
            "name": "grid_size",
            "type": "integer",
            "description": "Interpolation grid size (NxN). Default 256.",
            "required": False,
        },
        {
            "name": "test_title",
            "type": "string",
            "description": "Title prefix for the output plot (e.g., 'Bessel', 'Airy'). Default ''.",
            "required": False,
        },
        {
            "name": "file_path",
            "type": "string",
            "description": "Optional CSV path.",
            "required": False,
        },
        {
            "name": "n_xoy_slices",        # ← 新增参数
            "type": "integer",
            "description": "Number of XOY slices to plot in the extra slice figure. Default 16.",
            "required": False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            p = json.loads(params)
            z_max        = float(p.get("z_max", 0.5))
            z_steps      = int(p.get("z_steps", 50))
            N            = int(p.get("grid_size", 256))
            test_title   = p.get("test_title", "")
            file_path    = p.get("file_path") or get_layout_path()
            n_xoy_slices = int(p.get("n_xoy_slices", 16))   # ← 新增

            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}."

            df  = pd.read_csv(file_path)
            k   = CONSTANTS["k"]
            lam = CONSTANTS["lambda"]

            x_hex = df["x"].values
            y_hex = df["y"].values
            phase = df["target_phase"].values
            mag   = df["mag"].values if "mag" in df.columns else np.ones(len(df))

            margin      = CONSTANTS["p"] * 2
            x_min, x_max_v = x_hex.min() - margin, x_hex.max() + margin
            y_min, y_max_v = y_hex.min() - margin, y_hex.max() + margin

            xi = np.linspace(x_min, x_max_v, N)
            yi = np.linspace(y_min, y_max_v, N)
            XI, YI = np.meshgrid(xi, yi)

            from scipy.interpolate import griddata
            E_complex = mag * np.exp(1j * phase)
            E_real = griddata((x_hex, y_hex), E_complex.real, (XI, YI), method="linear", fill_value=0.0)
            E_imag = griddata((x_hex, y_hex), E_complex.imag, (XI, YI), method="linear", fill_value=0.0)
            E0 = E_real + 1j * E_imag

            dx = xi[1] - xi[0]
            dy = yi[1] - yi[0]
            fx = np.fft.fftfreq(N, d=dx)
            fy = np.fft.fftfreq(N, d=dy)
            FX, FY = np.meshgrid(fx, fy)
            kx  = 2 * np.pi * FX
            ky  = 2 * np.pi * FY
            kz2 = k**2 - kx**2 - ky**2
            kz  = np.where(kz2 >= 0, np.sqrt(kz2), 0)
            propagating_mask = (kz2 >= 0)
            A0  = np.fft.fft2(E0)

            z_arr = np.linspace(0.001, z_max, z_steps)
            intensity_slices = np.zeros((z_steps, N, N))
            for iz, z in enumerate(z_arr):
                H  = np.where(propagating_mask, np.exp(1j * kz * z), 0)
                Ez = np.fft.ifft2(A0 * H)
                intensity_slices[iz] = np.abs(Ez)**2

            # ================================================================
            # 图1：原始6格图（完全保留）
            # ================================================================
            fig = plt.figure(figsize=(20, 16))

            ax1 = fig.add_subplot(231)
            y_center_idx = N // 2
            xoz_slice = intensity_slices[:, y_center_idx, :]
            xoz_slice = xoz_slice / (xoz_slice.max() + 1e-9)
            ax1.imshow(
                xoz_slice,
                extent=[xi[0]*1e3, xi[-1]*1e3, z_arr[0]*1e3, z_arr[-1]*1e3],
                origin="lower", aspect="auto", cmap="hot"
            )
            ax1.set_xlabel("x (mm)")
            ax1.set_ylabel("z (mm)")
            ax1.set_title("XOZ Slice (y=0)\n← Airy trajectory here")
            plt.colorbar(ax1.images[0], ax=ax1, label="Norm. Intensity")

            ax2 = fig.add_subplot(232)
            x_center_idx = N // 2
            yoz_slice = intensity_slices[:, :, x_center_idx]
            yoz_slice = yoz_slice / (yoz_slice.max() + 1e-9)
            ax2.imshow(
                yoz_slice,
                extent=[yi[0]*1e3, yi[-1]*1e3, z_arr[0]*1e3, z_arr[-1]*1e3],
                origin="lower", aspect="auto", cmap="hot"
            )
            ax2.set_xlabel("y (mm)")
            ax2.set_ylabel("z (mm)")
            ax2.set_title("YOZ Slice (x=0)\n← Airy trajectory here")
            plt.colorbar(ax2.images[0], ax=ax2, label="Norm. Intensity")

            z_indices = [z_steps // 5, z_steps // 2, int(z_steps * 4 / 5)]
            z_labels  = ["z_near", "z_mid", "z_far"]
            for plot_idx, (zi, zlabel) in enumerate(zip(z_indices, z_labels)):
                ax = fig.add_subplot(233 + plot_idx)
                xoy      = intensity_slices[zi]
                xoy_norm = xoy / (xoy.max() + 1e-9)
                im = ax.imshow(
                    xoy_norm,
                    extent=[xi[0]*1e3, xi[-1]*1e3, yi[0]*1e3, yi[-1]*1e3],
                    origin="lower", cmap="hot", vmin=0, vmax=1
                )
                ax.set_xlabel("x (mm)")
                ax.set_ylabel("y (mm)")
                ax.set_title(f"XOY @ z={z_arr[zi]*1e3:.0f}mm ({zlabel})\n← Bessel rings here")
                plt.colorbar(im, ax=ax, label="Norm. Intensity")

            ax6 = fig.add_subplot(236)
            cx_arr = np.zeros(z_steps)
            cy_arr = np.zeros(z_steps)
            for iz in range(z_steps):
                sl    = intensity_slices[iz]
                total = sl.sum() + 1e-9
                cx_arr[iz] = (sl * XI).sum() / total * 1e3
                cy_arr[iz] = (sl * YI).sum() / total * 1e3
            ax6.plot(z_arr * 1e3, cx_arr, label="x centroid", color="red")
            ax6.plot(z_arr * 1e3, cy_arr, label="y centroid", color="blue")
            ax6.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax6.set_xlabel("z (mm)")
            ax6.set_ylabel("Centroid position (mm)")
            ax6.set_title("Main lobe trajectory\n← Airy: should curve parabolically")
            ax6.legend()
            ax6.grid(True, alpha=0.3)

            plt.suptitle(
                f"{test_title}  |  Near-field Propagation  |  z: 0 ~ {z_max*1e3:.0f}mm  |  λ={lam*1e3:.1f}mm",
                fontsize=13
            )
            plt.tight_layout()
            img_path = file_path.replace(".csv", "_nearfield.png")
            plt.savefig(img_path, dpi=150, bbox_inches="tight")
            plt.close()

            # ================================================================
            # 图2：多XOY截面图（新增）
            # ================================================================
            n_xoy_slices = min(n_xoy_slices, z_steps)   # 不超过总步数
            n_cols       = 4
            n_rows       = int(np.ceil(n_xoy_slices / n_cols))

            # 均匀采样 z 索引
            xoy_indices = np.linspace(0, z_steps - 1, n_xoy_slices, dtype=int)

            fig2, axes2 = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * 5, n_rows * 4.5)
            )
            axes2 = axes2.flatten()

            for plot_idx, zi in enumerate(xoy_indices):
                ax  = axes2[plot_idx]
                xoy = intensity_slices[zi]
                xoy_norm = xoy / (xoy.max() + 1e-9)
                im = ax.imshow(
                    xoy_norm,
                    extent=[xi[0]*1e3, xi[-1]*1e3, yi[0]*1e3, yi[-1]*1e3],
                    origin="lower", cmap="hot", vmin=0, vmax=1
                )
                ax.set_xlabel("x (mm)", fontsize=9)
                ax.set_ylabel("y (mm)", fontsize=9)
                ax.set_title(f"z = {z_arr[zi]*1e3:.0f} mm", fontsize=10)
                ax.axhline(0, color='cyan', linewidth=0.5, alpha=0.6)
                ax.axvline(0, color='cyan', linewidth=0.5, alpha=0.6)
                plt.colorbar(im, ax=ax, label="Norm. I", fraction=0.046, pad=0.04)

            # 隐藏多余子图
            for ax in axes2[n_xoy_slices:]:
                ax.set_visible(False)

            fig2.suptitle(
                f"{test_title}  |  XOY Slices ({n_xoy_slices} planes)  |  "
                f"z: {z_arr[xoy_indices[0]]*1e3:.0f}~{z_arr[xoy_indices[-1]]*1e3:.0f} mm  |  "
                f"λ={lam*1e3:.1f}mm",
                fontsize=13
            )
            plt.tight_layout()
            img_path2 = file_path.replace(".csv", "_nearfield_xoy_slices.png")
            plt.savefig(img_path2, dpi=150, bbox_inches="tight")
            plt.close()

            # ================================================================
            # 峰值位置
            # ================================================================
            peak_z_idx = np.array(
                [intensity_slices[iz].max() for iz in range(z_steps)]
            ).argmax()

            return (
                f"Near-field propagation done.\n"
                f" - Grid: {N}x{N}, z: 0~{z_max*1e3:.0f}mm ({z_steps} slices)\n"
                f" - Peak intensity @ z={z_arr[peak_z_idx]*1e3:.1f}mm\n"
                f" - 主图 (6格):     {img_path}\n"
                f" - XOY截面图 ({n_xoy_slices}张): {img_path2}"
            )

        except Exception as e:
            return f"Error: {traceback.format_exc()}"