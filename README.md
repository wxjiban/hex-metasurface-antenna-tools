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
| `radius` | number | 0.13 | 口面半径 (m) |
| `output_path` | string | current_layout.csv | 保存路径 |

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

### 4. calculate_pb_rotation - PB 相位

将目标相位转换为超表面单元旋转角度

| 参数 | 类型 | 说明 |
|------|------|------|
| `file_path` | string | CSV 文件路径 |

**转换公式：**
```
rotation_angle = target_phase / 2
mag = 0.000954 × target_phase + 0.9225
```

---

### 5. simulate_metasurface - 仿真验证

远场电磁仿真 + 结构可视化

| 参数 | 类型 | 说明 |
|------|------|------|
| `file_path` | string | CSV 文件路径 |

**输出：**
- `_layout.png` - 相位分布结构图
- `_farfield.png` - 3D 远场方向图

---

### 6. apply_collimate_lens - 准直透镜

球面波 → 平面波 转换

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

### 7. apply_axicon_phase - Bessel 波束

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

### 8. apply_airy_phase - Airy 波束

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

### 9. apply_beam_steering - 波束偏转

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

## 组合功能示例

| 组合 | 功能 | 模式 |
|------|------|------|
| 准直 + 偏转 | 球面波入射，平面波斜出射 |准直(overwrite) + 偏转(add) |
| 准直 + Bessel | 球面波入射，非衍射出射 |准直(overwrite) + Bessel(add) |

---

## 工作流程

```
1. generate_grid_layout()     → 初始化阵列
2. apply_xxx_phase()         → 施加相位 (可叠加)
3. calculate_pb_rotation()   → PB 相位转换
4. simulate_metasurface()    → 仿真验证
```

---

## 文件输出

- **CSV**: 单元坐标 + target_phase + rotation_angle_deg
- **PNG**: 结构图 + 远场方向图

默认路径: `outputs/current_layout.csv`

---

# metasurface #oam #bessel #airy #dammann #beam-steering #simulation
