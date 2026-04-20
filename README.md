# 超表面天线工具集 (Metasurface Antenna Tools)

> 基于 Qwen Agent 的超表面相位设计与仿真工具

---

## 物理常数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `freq` | 9×10⁹ Hz | 工作频率 (9 GHz) |
| `c` | 3×10⁸ m/s | 光速 |
| `p` (单元周期) | 7.2 mm | 相邻单元间距 |
| `lambda` | 0.0333 m | 自由空间波长 |
| `k` | 2π/λ | 波数 |

---

## 工具列表

### 1. generate_grid_layout - 阵列初始化

初始化超表面单元布局（六边形/矩形网格）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `layout_type` | string | "hex" | "hex" 或 "rect" |
| `radius` | number | 0.13 | 口面半径 (m)，决定单元数量 |
| `output_path` | string | current_layout.csv | 保存路径 |

**单元数量由 radius 控制：**
- radius=0.13m → 约 450 单元
- radius=0.08m → 约 180 单元

**示例：**
```json
{"layout_type": "hex", "radius": 0.13}
```

---

### 2. apply_vortex_phase - 涡旋波相位 (OAM)

施加拓扑涡旋相位，用于产生轨道角动量 (Orbital Angular Momentum)

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `charge_l` | integer | ✅ | 拓扑荷 (1, 2, -1 等) |
| `file_path` | string | - | CSV 文件路径 |

**相位公式：**
```
vortex_phase = l × arctan2(y, x)
```

---

### 3. apply_dammann_grating - 达曼光栅

NxN 光束分束器，适用于六边形晶格

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `beam_order` | integer | ✅ | 分束阶数 [2,3,4,5,6,7] |
| `period_multiple` | integer | 12 | 光栅周期倍数 |
| `file_path` | string | - | CSV 文件路径 |

**方法：** 离散晶格索引 (Axial Coordinates)

---

### 4. apply_collimate_lens - 球面波准直透镜

使用理论公式将球面波 → 平面波 转换

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `focal_length` | number | 0.15 | 焦距 (m) |
| `mode` | string | "overwrite" | "overwrite" 或 "add" |
| `file_path` | string | - | CSV 文件路径 |

**相位公式：**
```
phi = k × (√(x² + y² + f²) - f)
```

---

### 5. apply_axicon_phase - Bessel 波束

产生非衍射 Bessel 波束

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cone_angle_deg` | number | 10 | 锥角半角 (°) |
| `mode` | string | "overwrite" | "overwrite" 或 "add" |
| `file_path` | string | - | CSV 文件路径 |

**相位公式：**
```
phi = -k × sin(α) × √(x² + y²)
```

---

### 6. apply_airy_phase - Airy 波束

自加速 Airy 波束（三次相位调制）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cubic_coeff` | number | 5×10⁶ | 三次系数 (rad/m³) |
| `separable` | boolean | true | 是否分离二维 |
| `mode` | string | "overwrite" | "overwrite" 或 "add" |
| `file_path` | string | - | CSV 文件路径 |

**相位公式：**
```
phi = a3 × (x³ + y³)  [separable]
phi = a3 × ρ³         [radial]
```

---

### 7. apply_beam_steering - 波束偏转

线性梯度相位实现波束扫描

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `steer_theta_deg` | number | 30 | 俯仰角 (°) |
| `steer_phi_deg` | number | 0 | 方位角 (°) |
| `mode` | string | "overwrite" | "overwrite" 或 "add" |
| `file_path` | string | - | CSV 文件路径 |

**相位公式：**
```
phi = k × (x × sin(θ) × cos(φ) + y × sin(θ) × sin(φ))
```

---

### 8. configure_cp_efficiency - 圆极化配置

配置圆极化转换效率与旋向

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `eta` | number | 0.85 | 转换效率 (0~1) |
| `incident_pol` | string | "LCP" | "LCP" 或 "RCP" |
| `file_path` | string | - | CSV 文件路径 |

**说明：**
- LCP 入射：cp_sign = +1，相位 φ = +2θ，旋转 θ = φ/2
- RCP 入射：cp_sign = -1，相位 φ = -2θ，旋转 θ = -φ/2

---

### 9. apply_measured_compensation - 实测数据补偿

基于实测馈源相位/幅度数据的补偿（核心功能）

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `phase_csv` | string | ✅ | 实测相位CSV (23×23，mm单位) |
| `magnitude_csv` | string | - | 实测幅度CSV (同格式) |
| `additional_phase_csv` | string | - | 额外叠加的相位CSV |
| `file_path` | string | - | 阵列布局CSV |

**流程：**
1. 解析实测CSV（23×23网格，坐标mm）
2. 线性插值到六边形单元坐标
3. 计算补偿相位 = 2π - measured_phase
4. 可选叠加额外相位（如偏转、涡旋）

**输出：**
- `measured_phase` - 实测入射相位
- `measured_mag` - 实测幅度
- `compensation_phase` - 补偿相位
- `target_phase` - 最终目标相位

---

### 10. calculate_pb_rotation - PB 相位转换

将目标相位转换为超表面单元旋转角度（PB机制）

| 参数 | 类型 | 说明 |
|------|------|------|
| `file_path` | string | CSV 文件路径 |

**转换公式：**
```
LCP: rotation_angle = target_phase / 2
RCP: rotation_angle = -target_phase / 2

mag = measured_mag (如果有实测数据)
mag = 0.000954 × target_phase + 0.9225 (线性近似)
```

---

### 11. simulate_metasurface - 远场仿真

CP-aware 远场电磁仿真 + 结构可视化

| 参数 | 类型 | 说明 |
|------|------|------|
| `file_path` | string | CSV 文件路径 |

**输出：**
- `_layout.png` - 六边形结构图（相位+幅度分布）
- `_incident_vs_comp.png` - 入射相位与补偿相位对比图
- `_farfield.png` - 3D 远场方向图（Cross-pol / Co-pol / Total）

---

## 实测数据文件格式

4个CSV文件（LCP/RCP 各相位+幅度）：

| 文件名 | 内容 | 格式 |
|--------|------|------|
| `Left_circular_Phase_mm.csv` | 左旋相位 | 23×23网格，坐标mm，相位rad |
| `Left_circular_Magnitude_mm.csv` | 左旋幅度 | 同上，幅度值 |
| `Right_circular_Phase_mm.csv` | 右旋相位 | 同上 |
| `Right_circular_Magnitude_mm.csv` | 右旋幅度 | 同上 |

CSV结构：
```
     -81mm  -72mm  -63mm  ...  72mm   81mm
-81mm  0.12   0.15   0.18  ...  0.14   0.11
-72mm  0.14   0.17   0.20  ...  0.16   0.13
...
```

---

## 组合功能示例

| 组合 | 功能 | 模式 |
|------|------|------|
| 准直 | 球面波入射，平面波出射 | 理论公式 overwrite |
| 准直 + 偏转 | 平面波斜出射 | 准直(overwrite) + 偏转(add) |
| 准直 + Bessel | 非衍射出射 | 准直(overwrite) + Bessel(add) |
| 实测补偿 | 基于实测数据的准直 | apply_measured_compensation |
| 实测补偿 + 偏转 | 补偿后波束扫描 | 补偿 + 偏转(add) |

---

## 工作流程

### 流程A：理论公式准直
```
1. generate_grid_layout()     → 初始化阵列
2. apply_collimate_lens()    → 理论公式准直
3. calculate_pb_rotation()   → PB相位移转
4. simulate_metasurface()    → 仿真验证
```

### 流程B：实测数据补偿
```
1. generate_grid_layout()     → 初始化阵列
2. configure_cp_efficiency()  → 配置CP参数
3. apply_measured_compensation() → 加载实测数据补偿
4. calculate_pb_rotation()    → PB相位移转
5. simulate_metasurface()     → 仿真验证
```

---

## 文件输出

**输出目录：** `outputs/`

**CSV文件：**
- `current_layout.csv` - 单元坐标 + 各类相位 + 旋转角

**PNG文件：**
- `current_layout_layout.png` - 结构图
- `current_layout_incident_vs_comp.png` - 入射/补偿相位对比
- `current_layout_farfield.png` - 3D远场图

**自动备份：** 运行前自动备份到 `outputs_backup/backup_YYYYMMDD_HHMMSS/`

---

## 测试脚本

`test_new_functions.py` 提供多种测试场景：

```python
test_lcp_collimate()          # A: LCP实测补偿 → 平面波
test_rcp_collimate()          # B: RCP实测补偿 → 平面波
test_lcp_collimate_steer()    # C: 补偿 + 波束偏转
test_lcp_collimate_bessel()   # D: 补偿 + Bessel波束
test_lcp_collimate_airy()     # E: 补偿 + Airy波束
test_lcp_collimate_dammann()  # F: 补偿 + 达曼光栅
```

---

# metasurface #oam #bessel #airy #dammann #beam-steering #simulation #measured-feed