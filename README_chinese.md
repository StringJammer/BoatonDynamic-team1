# Photometric Stereo (光度立体视觉) 项目

> 基于5光源的光度立体视觉与深度重建系统

## 📖 项目简介

本项目实现了一个完整的光度立体视觉系统，通过分析物体在不同光照条件下的5张输入图像，重建物体的表面法向量、反照率图和深度图。系统支持多种材质物体的三维重建，包括石膏几何体和泡沫塑料物体。

### 核心功能

- ✅ **表面法向量重建** - 从多幅图像计算每个像素的表面方向
- ✅ **反照率图生成** - 提取物体的固有颜色/反射率
- ✅ **深度图重建** - 使用 Frankot-Chellappa 或 Poisson 方法恢复3D形状
- ✅ **多场景支持** - 支持不同材质和形状的物体
- ✅ **高性能处理** - 多线程并行计算，支持大规模图像处理

## 🏗️ 项目结构

```
images and code/
├── photometric stereo(for 's' datasets).py  # 主程序
├── readme.txt                                # 数据集说明
├── images_s1/                                # S1 场景图像（石膏六面体）
├── images_s2/                                # S2 场景图像（石膏圆锥）
├── images_s3/                                # S3 场景图像（泡沫塑料苹果）
├── results_s1/                               # S1 处理结果
├── results_s2/                               # S2 处理结果
└── results_s3/                               # S3 处理结果
```

## 📊 数据集说明

### 场景描述

| 场景 | 物体材质 | 形状 |
|------|---------|------|
| S1 | 石膏 (Gypsum) | 六面体 (Hexahedron) |
| S2 | 石膏 (Gypsum) | 圆锥 (Triangular Cone) |
| S3 | 泡沫塑料 (Foamed Plastics) | 苹果 (Apple) |

### 光源配置

5个光源的方向向量（归一化后）：

```
l1: [-35.3,  35.3, 50]  →  [-0.500,  0.500, 0.707]
l2: [  0.0,  50.0, 50]  →  [ 0.000,  0.707, 0.707]
l3: [ 50.0,   0.0, 50]  →  [ 0.707,  0.000, 0.707]
l4: [  0.0, -50.0, 50]  →  [ 0.000, -0.707, 0.707]
l5: [-35.3, -35.3, 50]  →  [-0.500, -0.500, 0.707]
```

### 相机参数

- **镜头**：50mm
- **光圈**：f/10
- **位置**：[-60, 0, 25] cm

## 🚀 快速开始

### 环境要求

```bash
numpy>=1.20.0
opencv-python>=4.5.0
scipy>=1.7.0
```

### 安装依赖

```bash
pip install numpy opencv-python scipy
```

### 运行程序

编辑 `photometric stereo(for 's' datasets).py` 文件，修改配置参数：

```python
SCENE_NAME = "s1"  # 选择场景: "s1", "s2" 或 "s3"
SCALE_FACTOR = 0.25  # 图像缩放因子 (0-1)
```

然后运行：

```python
python "photometric stereo(for 's' datasets).py"
```

## 📈 处理流程

### 1. 光度立体视觉 (Photometric Stereo)

```
5张输入图像 → 预处理 → 光照矩阵求解 → 法向量计算 → 反照率提取
```

**步骤详情：**
- 图像加载与裁剪（去除边缘10%）
- 图像缩放（默认25%）
- 背景过滤（保留物体区域）
- 多线程并行计算法向量和反照率

### 2. 深度图重建 (Depth Reconstruction)

```
法向量图 → 梯度计算 → 积分重建 → 深度归一化
```

**支持方法：**
- **Frankot-Chellappa 方法**：频域全局优化，速度快，精度高
- **Poisson 方法**：时域迭代求解，可调整迭代次数

## 📁 输出文件

每个场景处理后生成以下文件：

### 可视化图像

| 文件名 | 说明 |
|--------|------|
| `normal_map_s{N}.png` | 法向量图可视化 |
| `albedo_map_s{N}.png` | 反照率图 |
| `depth_map_s{N}_frankot_chellappa.png` | 深度图（Frankot-Chellappa方法） |

### 原始数据

| 文件名 | 格式 | 说明 |
|--------|------|------|
| `normal_map_float_s{N}.npy` | NumPy | 法向量浮点数据 |
| `albedo_map_float_s{N}.npy` | NumPy | 反照率浮点数据 |
| `depth_raw_s{N}_frankot_chellappa.npy` | NumPy | 深度原始数据 |
| `mask_valid_s{N}.npy` | NumPy | 有效区域掩码 |
| `light_matrix_s{N}.txt` | 文本 | 光源方向矩阵 |

## 🔧 技术细节

### 光度立体视觉原理

对于朗伯体表面，每个像素的亮度满足：

```
I = ρ · (n · L)
```

其中：
- `I` - 观测强度
- `ρ` - 反照率（表面反射率）
- `n` - 单位表面法向量
- `L` - 单位光源方向向量

对于5个光源，可以建立线性方程组：

```
I₁ = ρ · (n · L₁)
I₂ = ρ · (n · L₂)
I₃ = ρ · (n · L₃)
I₄ = ρ · (n · L₄)
I₅ = ρ · (n · L₅)
```

通过最小二乘法求解可得法向量和反照率。

### 深度重建方法

#### Frankot-Chellappa 方法

在频域求解泊松方程：

```
∇²Z = ∇ · (-nₓ/nz, -nᵧ/nz)
```

优点：全局最优解，计算速度快

#### Poisson 方法

时域迭代求解：

```
Z[i,j] = (Z[i-1,j] + Z[i+1,j] + Z[i,j-1] + Z[i,j+1] - div[i,j]) / 4
```

优点：灵活性高，可控制迭代精度

## 📊 性能优化

- **多线程处理**：使用 `ThreadPoolExecutor` 并行计算像素
- **块处理**：避免大图像内存溢出
- **进度显示**：实时显示处理进度
- **背景过滤**：减少无效计算

## 🎯 示例结果

### S1 - 石膏六面体

**法向量图**
<img src="images and code/results_s1/normal_map_s1.png" width="100%" />

**反照率图**
<img src="images and code/results_s1/albedo_map_s1.png" width="100%" />

**深度图**
<img src="images and code/results_s1/depth_map_s1_frankot_chellappa.png" width="100%" />

### S2 - 石膏圆锥

**法向量图**
<img src="images and code/results_s2/normal_map_s2.png" width="100%" />

**反照率图**
<img src="images and code/results_s2/albedo_map_s2.png" width="100%" />

**深度图**
<img src="images and code/results_s2/depth_map_s2_frankot_chellappa.png" width="100%" />

### S3 - 泡沫塑料苹果

**法向量图**
<img src="images and code/results_s3/normal_map_s3.png" width="100%" />

**反照率图**
<img src="images and code/results_s3/albedo_map_s3.png" width="100%" />

**深度图**
<img src="images and code/results_s3/depth_map_s3_frankot_chellappa.png" width="100%" />

## 📝 注意事项

1. **图像要求**：确保输入图像命名格式为 `img1.jpg` 到 `img5.jpg`
2. **内存管理**：处理大图像时适当调整 `SCALE_FACTOR`
3. **背景过滤**：程序会自动过滤背景，但确保物体与背景有明显对比
4. **光源方向**：修改光源矩阵需确保向量正确归一化

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目仅用于学习和研究目的。

## 🔗 参考文献

- Woodham, R. J. (1980). "Photometric method for determining surface orientation from multiple images."
- Frankot, R. T., & Chellappa, R. (1988). "A method for enforcing integrability in shape from shading algorithms."

---

**作者**: JulianZhu  
**邮箱**: 1141911921@qq.com
