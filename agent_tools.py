import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from qwen_agent.tools.base import BaseTool, register_tool

# ================= 全局配置 =================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
print(PROJECT_ROOT)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 【关键修改】定义默认文件路径
DEFAULT_LAYOUT_PATH = os.path.join(OUTPUT_DIR, "current_layout.csv")

CONSTANTS = {
    "freq": 9e9,
    "c": 3e8,
    "p": 7.2e-3,  # 单元周期
    "lambda": 3e8 / 9e9,
    "k": 2 * math.pi / (3e8 / 9e9)
}

def _phase_normalize(phase):
    return phase % (2 * np.pi)

# ================= Tool 0: 阵列初始化 =================

@register_tool('generate_grid_layout')
class GenerateGridLayout(BaseTool):
    description = (
        'Initialize the metasurface array layout. '
        'Call this FIRST to create the coordinate system. '
        'If no output_path is provided, saves to default location.'
    )
    
    parameters = [{
        'name': 'layout_type',
        'type': 'string',
        'description': 'Type of grid: "hex" or "rect". Default "hex".',
        'required': False
    }, {
        'name': 'radius',
        'type': 'number',
        'description': 'Aperture radius in meters. Default 0.13.',
        'required': False
    }, {
        'name': 'output_path',
        'type': 'string',
        'description': f'Optional save path. Default: {DEFAULT_LAYOUT_PATH}',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json
            p = json.loads(params)
            layout_type = p.get('layout_type', 'hex')
            radius = float(p.get('radius', 0.13))
            # 【逻辑修改】使用默认路径
            save_path = p.get('output_path') or DEFAULT_LAYOUT_PATH
            
            period = CONSTANTS["p"]
            xyset = set()
            
            if layout_type == 'hex':
                x, y = period / 2, 0 
                xyset.add((x, y))
                steps = math.ceil(radius / period)
                for _ in range(steps):
                    new_points = set()
                    for cx, cy in xyset:
                        angles = [0, 60, 120, 180, 240, 300]
                        for ang in angles:
                            rad = math.radians(ang)
                            nx = round(cx + period * math.cos(rad), 10)
                            ny = round(cy + period * math.sin(rad), 10)
                            if nx**2 + ny**2 < radius**2:
                                new_points.add((nx, ny))
                    xyset |= new_points
            
            df = pd.DataFrame(list(xyset), columns=['x', 'y'])
            df['target_phase'] = 0.0 
            df['rotation_angle_deg'] = 0.0
            
            df.to_csv(save_path, index=False)
            return f"Layout initialized ({layout_type}). Total Units: {len(df)}. Saved to: {save_path}"
        except Exception as e:
            return f"Error: {str(e)}"

# ================= Tool 1: 涡旋波理论 =================

@register_tool('apply_vortex_phase')
class ApplyVortexPhase(BaseTool):
    description = 'Apply Vortex Phase (OAM) to the layout.'
    
    parameters = [{
        'name': 'charge_l',
        'type': 'integer',
        'description': 'Topological charge (e.g., 1, 2, -1).',
        'required': True
    }, {
        'name': 'file_path',
        'type': 'string',
        'description': 'Optional path to CSV.',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json
            import pandas as pd
            import numpy as np
            
            p = json.loads(params)
            l = int(p.get('charge_l', 1))
            file_path = p.get('file_path') or DEFAULT_LAYOUT_PATH
            
            if not os.path.exists(file_path): 
                return f"Error: File {file_path} not found."
            
            df = pd.read_csv(file_path)
            
            # 计算涡旋相位 (arctan2 返回 -pi 到 pi)
            # 加上 1e-9 防止 (0,0) 点报错
            theta = np.arctan2(df['y'], df['x'] + 1e-9)
            vortex_phase = l * theta
            
            # 更新 target_phase
            # 如果之前有相位（比如透镜相位），则是叠加；如果没有，就是覆盖
            if 'target_phase' not in df.columns:
                df['target_phase'] = 0.0
            
            # 这里选择【覆盖】还是【叠加】？
            # 通常如果是纯涡旋测试，建议覆盖；如果是透镜叠加涡旋，用 +=
            # 为了保险，这里假设是叠加到基础相位（0）上
            df['target_phase'] += vortex_phase
            
            # 【关键修复】手动归一化到 [0, 2pi]
            df['target_phase'] = df['target_phase'] % (2 * np.pi)
            
            df.to_csv(file_path, index=False)
            return f"Vortex Phase (l={l}) applied to {len(df)} units."
        except Exception as e:
            return f"Error: {str(e)}"

# ================= Tool 2: 达曼光栅理论 =================
 
@register_tool('apply_dammann_grating')
class ApplyDammannGrating(BaseTool):
    description = (
        'Apply Dammann Grating phase for NxN beam splitting adapted for Hexagonal Grid. '
        'Uses discrete lattice indices (Axial Coordinates) to ensure sharp phase transitions.'
    )
    
    parameters = [
        {
            'name': 'file_path',
            'type': 'string',
            'description': 'Optional path to CSV. Default is the current active layout.',
            'required': False
        },
        {
            'name': 'beam_order',
            'type': 'integer',
            'description': 'Number of beams in one dimension. Supports [2, 3, 4, 5, 6, 7].',
            'required': True 
        },
        {
            'name': 'period_multiple',
            'type': 'integer',
            'description': 'The grating period as a multiple of unit cell period "p". Default is 12.',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json
            import pandas as pd
            import numpy as np
            
            p_dict = json.loads(params)
            file_path = p_dict.get('file_path') or DEFAULT_LAYOUT_PATH
            
            # 1. 获取参数
            N = int(p_dict.get('beam_order', 5)) 
            M = int(p_dict.get('period_multiple', 12)) 
            unit_p = CONSTANTS["p"]
            
            if not os.path.exists(file_path): 
                return f"Error: File not found at {file_path}."
            
            df = pd.read_csv(file_path)

            # =========================================================
            # 2. 坐标转换：物理坐标 (x,y) -> 晶格整数索引 (q, r)
            # 使用 Axial Coordinates 确保相位边缘完美切合六边形单元
            # =========================================================
            sqrt3 = np.sqrt(3)
            
            # r: 行索引 (Row Index), 沿着 Y 轴方向
            # 公式: y = r * (sqrt(3)/2 * p)
            r_float = 2 * df['y'].values / (sqrt3 * unit_p)
            r = np.round(r_float).astype(int)
            
            # q: 斜向列索引 (Diagonal Column Index)
            # 公式: x = (q + r/2) * p
            q_float = df['x'].values / unit_p - r_float / 2
            q = np.round(q_float).astype(int)
            
            # 定义光栅的两个传播轴 (u, v)
            # 对应晶格的两个自然生长方向
            u_idx = q
            v_idx = r

            # =========================================================
            # 3. 达曼跳变点配置 (全周期展开逻辑)
            # =========================================================
            # 原始半周期点 [0, 0.5]
            raw_transitions = {
                2: [0.5],
                4: [0.1767, 0.4577],
                6: [0.1084, 0.2941, 0.4519],
                3: [0.2656],
                5: [0.1322, 0.4808],
                7: [0.0886, 0.3060, 0.4495]
            }

            if N not in raw_transitions:
                return f"Error: Beam order {N} not supported."
            
            raw_points = raw_transitions[N]
            
            # 展开为全周期 [0, 1] 的点，修复 N=2 时无法翻转的 Bug
            if N == 2:
                # N=2 特例: 0.5 对称后还是 0.5
                full_points = [0.5]
            else:
                # 对称展开: P U {1-p}
                full_points = sorted(list(set(raw_points + [1.0 - pt for pt in raw_points])))

            # =========================================================
            # 4. 核心相位计算
            # =========================================================
            def get_phase_from_index(index_array, period_int, pts):
                # 归一化到 [0, 1)
                # 使用整数取模，精度完全可控
                norm = (index_array % period_int) / period_int
                
                # 统计跳变次数
                flip_counts = np.zeros_like(norm, dtype=int)
                for pt in pts:
                    # 判断当前位置是否超过跳变点
                    flip_counts += (norm > pt).astype(int)
                
                # 奇数次翻转 -> pi, 偶数次 -> 0
                return np.where(flip_counts % 2 != 0, np.pi, 0.0)

            # 计算两个方向的相位分量
            phi_u = get_phase_from_index(u_idx, M, full_points)
            phi_v = get_phase_from_index(v_idx, M, full_points)
            
            # 叠加
            total_dammann_phase = phi_u + phi_v
            
            # =========================================================
            # 5. 保存结果
            # =========================================================
            if 'target_phase' not in df.columns: df['target_phase'] = 0.0
            
            df['target_phase'] += total_dammann_phase
            
            # 最终归一化到 [0, 2pi]
            df['target_phase'] = df['target_phase'] % (2 * np.pi)
            
            df.to_csv(file_path, index=False)
            
            return (f"Success: Applied {N}x{N} Hexagonal Dammann Grating.\n"
                    f" - Method: Discrete Lattice Indexing (Bug fixed)\n"
                    f" - Period M: {M} units\n"
                    f" - Transition Points (Full Period): {np.round(full_points, 4)}")
            
        except Exception as e:
            import traceback
            return f"Error: {str(e)}\n{traceback.format_exc()}"
# ================= Tool 3: PB 相位理论 =================

@register_tool('calculate_pb_rotation')
class CalculatePBRotation(BaseTool):
    description = (
        'Apply PB Phase theory. Converts Target Phase -> Rotation Angle. '
        'Formula: Theta = Phi / 2.'
    )
    
    parameters = [{
        'name': 'file_path',
        'type': 'string',
        'description': 'Optional path to CSV.',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json
            p = json.loads(params)
            file_path = p.get('file_path') or DEFAULT_LAYOUT_PATH
            
            if not os.path.exists(file_path): return f"Error: File not found at {file_path}."
            df = pd.read_csv(file_path)
            
            if 'target_phase' not in df.columns:
                return "Error: No 'target_phase' found."
            
            # PB 转换
            df['rotation_angle_rad'] = df['target_phase'] / 2.0
            df['rotation_angle_deg'] = np.degrees(df['rotation_angle_rad'])
            
            # 模拟幅度
            intercept, slope = 0.000954, 0.9225
            df['mag'] = df['target_phase'] * intercept + slope
            
            df.to_csv(file_path, index=False)
            return f"PB Rotation calculated. Updated {file_path}"
        except Exception as e:
            return f"Error: {str(e)}"

# ================= Tool 4: 仿真验证 =================

@register_tool('simulate_metasurface')
class SimulateMetasurface(BaseTool):
    description = (
        'Run Far-field EM simulation and visualize structure. '
        'Uses the default layout file if no path is provided.'
    )
    
    parameters = [{
        'name': 'file_path',
        'type': 'string',
        'description': 'Optional path to CSV.',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json
            p = json.loads(params)
            file_path = p.get('file_path') or DEFAULT_LAYOUT_PATH
            
            if not os.path.exists(file_path): return f"Error: File not found at {file_path}."
            
            df = pd.read_csv(file_path)
            xy = df[['x', 'y']].values
            
            # 必须用 target_phase 画结构图
            phase_data = df['target_phase'].values if 'target_phase' in df.columns else np.zeros(len(df))
            mags = df['mag'].values if 'mag' in df.columns else np.ones(len(df))
            
            # --- 1. 画结构图 (PatchCollection) ---
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect('equal')
            radius = CONSTANTS['p'] / math.sqrt(3)
            
            patches = []
            colors_list = []
            norm = colors.Normalize(vmin=0, vmax=2 * np.pi)
            cmap = plt.get_cmap('jet')
            
            for idx, (x, y) in enumerate(xy):
                color = cmap(norm(phase_data[idx]))
                poly = RegularPolygon((x, y), numVertices=6, radius=radius, orientation=np.radians(0))
                patches.append(poly)
                colors_list.append(color)
            
            p_collection = PatchCollection(patches, facecolors=colors_list, edgecolors='black', linewidths=0.1)
            ax.add_collection(p_collection)
            ax.autoscale_view()
            ax.axis('off')
            ax.set_title("Metasurface Phase Distribution")
            
            img_struct = file_path.replace(".csv", "_layout.png")
            plt.savefig(img_struct, dpi=300, bbox_inches='tight')
            plt.close()
            
            # --- 2. 远场仿真 ---
            k = CONSTANTS["k"]
            M, N = 80, 80 
            theta = np.linspace(0, np.radians(60), M) 
            phi = np.linspace(0, 2*np.pi, N)
            THETA, PHI = np.meshgrid(theta, phi)
            u = np.sin(THETA) * np.cos(PHI)
            v = np.sin(THETA) * np.sin(PHI)
            
            E_total = np.zeros_like(THETA, dtype=complex)
            X, Y = xy[:, 0], xy[:, 1]
            
            batch_size = 200
            for i in range(0, len(X), batch_size):
                end = min(i + batch_size, len(X))
                x_b = X[i:end][:, None, None]
                y_b = Y[i:end][:, None, None]
                p_b = phase_data[i:end][:, None, None]
                m_b = mags[i:end][:, None, None]
                path_diff = k * (x_b * u + y_b * v)
                E_total += np.sum(m_b * np.exp(1j * (p_b + path_diff)), axis=0)
            
            E_norm = np.abs(E_total)
            E_norm = E_norm / (np.max(E_norm) + 1e-9)
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            X_plt = E_norm * np.sin(THETA) * np.cos(PHI)
            Y_plt = E_norm * np.sin(THETA) * np.sin(PHI)
            Z_plt = E_norm * np.cos(THETA)
            
            surf = ax.plot_surface(X_plt, Y_plt, Z_plt, cmap='jet', rcount=60, ccount=60, alpha=0.9)
            fig.colorbar(surf, shrink=0.5, aspect=10)
            
            img_farfield = file_path.replace(".csv", "_farfield.png")
            plt.savefig(img_farfield, dpi=150)
            plt.close()
            
            return f"Simulation Done.\n1. Structure: {img_struct}\n2. Far-field: {img_farfield}"
            
        except Exception as e:
            import traceback
            return f"Sim Error: {traceback.format_exc()}"
        
# ================= Tool 5: 球面波→平面波 (准直透镜) =================

@register_tool('apply_collimate_lens')
class ApplyCollimateLens(BaseTool):
    description = (
        'Convert a spherical wave from a point source into a plane wave (collimation). '
        'Applies a compensating lens phase: phi = k*(sqrt(x^2+y^2+f^2) - f) to cancel '
        'the spherical wavefront curvature. The point source is at distance f below the surface.'
    )

    parameters = [
        {
            'name': 'focal_length',
            'type': 'number',
            'description': 'Distance from point source to metasurface center (meters). Default 0.15.',
            'required': False
        },
        {
            'name': 'mode',
            'type': 'string',
            'description': '"overwrite" to replace existing phase, "add" to superpose. Default "overwrite".',
            'required': False
        },
        {
            'name': 'file_path',
            'type': 'string',
            'description': 'Optional path to CSV.',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json
            p = json.loads(params)
            f = float(p.get('focal_length', 0.15))
            mode = p.get('mode', 'overwrite')
            file_path = p.get('file_path') or DEFAULT_LAYOUT_PATH

            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}."

            df = pd.read_csv(file_path)
            k = CONSTANTS["k"]
            x = df['x'].values
            y = df['y'].values

            # 球面波在口面上的相位分布为 -k*sqrt(x^2+y^2+f^2)
            # 要补偿为平面波(相位=常数), 需要施加:
            #   phi_comp = k * (sqrt(x^2 + y^2 + f^2) - f)
            # 这样总输出相位 = -k*sqrt(...) + k*(sqrt(...)-f) = -k*f (常数) → 平面波
            r_sq = x**2 + y**2
            collimate_phase = k * (np.sqrt(r_sq + f**2) - f)

            if 'target_phase' not in df.columns:
                df['target_phase'] = 0.0

            if mode == 'overwrite':
                df['target_phase'] = collimate_phase
            else:
                df['target_phase'] += collimate_phase

            df['target_phase'] = df['target_phase'] % (2 * np.pi)
            df.to_csv(file_path, index=False)

            max_phase = np.max(collimate_phase)
            return (f"Collimating Lens applied (f={f:.4f}m, k={k:.2f}).\n"
                    f" - Max compensating phase: {max_phase:.2f} rad ({np.degrees(max_phase):.1f}°)\n"
                    f" - Mode: {mode}\n"
                    f" - Units: {len(df)}")
        except Exception as e:
            import traceback
            return f"Error: {traceback.format_exc()}"


# ================= Tool 6: Bessel非衍射波束 (轴棱锥) =================

@register_tool('apply_axicon_phase')
class ApplyAxiconPhase(BaseTool):
    description = (
        'Generate a Bessel (non-diffracting) beam by applying an axicon-like linear radial phase. '
        'Phase profile: phi = -k * sin(alpha) * sqrt(x^2+y^2), where alpha is the cone half-angle. '
        'Bessel beams resist diffraction over a long propagation distance.'
    )

    parameters = [
        {
            'name': 'cone_angle_deg',
            'type': 'number',
            'description': 'Axicon cone half-angle in degrees. Controls beam radius. Default 10.',
            'required': False
        },
        {
            'name': 'mode',
            'type': 'string',
            'description': '"overwrite" or "add". Default "overwrite".',
            'required': False
        },
        {
            'name': 'file_path',
            'type': 'string',
            'description': 'Optional path to CSV.',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json
            p = json.loads(params)
            alpha_deg = float(p.get('cone_angle_deg', 10.0))
            mode = p.get('mode', 'overwrite')
            file_path = p.get('file_path') or DEFAULT_LAYOUT_PATH

            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}."

            df = pd.read_csv(file_path)
            k = CONSTANTS["k"]
            alpha = np.radians(alpha_deg)
            x = df['x'].values
            y = df['y'].values

            rho = np.sqrt(x**2 + y**2)
            axicon_phase = -k * np.sin(alpha) * rho

            if 'target_phase' not in df.columns:
                df['target_phase'] = 0.0

            if mode == 'overwrite':
                df['target_phase'] = axicon_phase
            else:
                df['target_phase'] += axicon_phase

            df['target_phase'] = df['target_phase'] % (2 * np.pi)
            df.to_csv(file_path, index=False)

            z_max = rho.max() / np.tan(alpha) if np.tan(alpha) > 1e-9 else float('inf')
            return (f"Axicon (Bessel beam) phase applied.\n"
                    f" - Cone half-angle: {alpha_deg}°\n"
                    f" - Estimated non-diffracting range: {z_max:.4f} m\n"
                    f" - Mode: {mode}, Units: {len(df)}")
        except Exception as e:
            import traceback
            return f"Error: {traceback.format_exc()}"


# ================= Tool 7: Airy自加速波束 (三次相位) =================

@register_tool('apply_airy_phase')
class ApplyAiryPhase(BaseTool):
    description = (
        'Generate a self-accelerating Airy beam via cubic phase modulation. '
        'The beam follows a parabolic trajectory in free space without external force. '
        'Phase: phi = a3*(kx^3 + ky^3), applied in spectral domain approximation on the aperture.'
    )

    parameters = [
        {
            'name': 'cubic_coeff',
            'type': 'number',
            'description': 'Cubic phase coefficient (rad/m^3). Controls acceleration. Default 5e6.',
            'required': False
        },
        {
            'name': 'separable',
            'type': 'boolean',
            'description': 'If true, apply separable 1D Airy in x and y independently (2D Airy). Default true.',
            'required': False
        },
        {
            'name': 'mode',
            'type': 'string',
            'description': '"overwrite" or "add". Default "overwrite".',
            'required': False
        },
        {
            'name': 'file_path',
            'type': 'string',
            'description': 'Optional path to CSV.',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json
            p = json.loads(params)
            a3 = float(p.get('cubic_coeff', 5e6))
            separable = p.get('separable', True)
            mode = p.get('mode', 'overwrite')
            file_path = p.get('file_path') or DEFAULT_LAYOUT_PATH

            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}."

            df = pd.read_csv(file_path)
            x = df['x'].values
            y = df['y'].values

            if separable:
                airy_phase = a3 * (x**3 + y**3)
            else:
                # 径向三次相位 (圆对称Airy)
                rho = np.sqrt(x**2 + y**2)
                airy_phase = a3 * rho**3

            if 'target_phase' not in df.columns:
                df['target_phase'] = 0.0

            if mode == 'overwrite':
                df['target_phase'] = airy_phase
            else:
                df['target_phase'] += airy_phase

            df['target_phase'] = df['target_phase'] % (2 * np.pi)
            df.to_csv(file_path, index=False)

            return (f"Airy beam cubic phase applied.\n"
                    f" - Coefficient a3: {a3:.2e} rad/m³\n"
                    f" - Separable 2D: {separable}\n"
                    f" - Mode: {mode}, Units: {len(df)}")
        except Exception as e:
            import traceback
            return f"Error: {traceback.format_exc()}"


# ================= Tool 8: 波束偏转 (线性梯度相位) =================

@register_tool('apply_beam_steering')
class ApplyBeamSteering(BaseTool):
    description = (
        'Steer the transmitted beam to an arbitrary direction (theta, phi) using a linear phase gradient. '
        'Phase: phi(x,y) = k*(x*sin(theta)*cos(phi) + y*sin(theta)*sin(phi)). '
        'Can be combined with other phase profiles for simultaneous focusing + steering.'
    )

    parameters = [
        {
            'name': 'steer_theta_deg',
            'type': 'number',
            'description': 'Elevation steering angle in degrees from broadside. Default 30.',
            'required': False
        },
        {
            'name': 'steer_phi_deg',
            'type': 'number',
            'description': 'Azimuth steering angle in degrees. Default 0 (steer in xz-plane).',
            'required': False
        },
        {
            'name': 'mode',
            'type': 'string',
            'description': '"overwrite" or "add". Default "overwrite".',
            'required': False
        },
        {
            'name': 'file_path',
            'type': 'string',
            'description': 'Optional path to CSV.',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            import json
            p = json.loads(params)
            theta_d = float(p.get('steer_theta_deg', 30.0))
            phi_d = float(p.get('steer_phi_deg', 0.0))
            mode = p.get('mode', 'overwrite')
            file_path = p.get('file_path') or DEFAULT_LAYOUT_PATH

            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}."

            df = pd.read_csv(file_path)
            k = CONSTANTS["k"]
            theta = np.radians(theta_d)
            phi = np.radians(phi_d)
            x = df['x'].values
            y = df['y'].values

            steer_phase = k * (x * np.sin(theta) * np.cos(phi) +
                               y * np.sin(theta) * np.sin(phi))

            if 'target_phase' not in df.columns:
                df['target_phase'] = 0.0

            if mode == 'overwrite':
                df['target_phase'] = steer_phase
            else:
                df['target_phase'] += steer_phase

            df['target_phase'] = df['target_phase'] % (2 * np.pi)
            df.to_csv(file_path, index=False)

            return (f"Beam steering phase applied.\n"
                    f" - Direction: θ={theta_d}°, φ={phi_d}°\n"
                    f" - Phase gradient: kx={k*np.sin(theta)*np.cos(phi):.2f}, "
                    f"ky={k*np.sin(theta)*np.sin(phi):.2f} rad/m\n"
                    f" - Mode: {mode}, Units: {len(df)}")
        except Exception as e:
            import traceback
            return f"Error: {traceback.format_exc()}"