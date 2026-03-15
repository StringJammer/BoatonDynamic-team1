# Photometric Stereo Project

> 5-Light Source Photometric Stereo and Depth Reconstruction System

## 📖 Project Introduction

This project implements a complete photometric stereo system that reconstructs surface normal vectors, albedo maps, and depth maps of objects by analyzing 5 input images under different lighting conditions. The system supports 3D reconstruction of various material objects, including gypsum geometries and foam plastic objects.

### Core Features

- ✅ **Surface Normal Reconstruction** - Calculate surface orientation for each pixel from multiple images
- ✅ **Albedo Map Generation** - Extract intrinsic color/reflectance of objects
- ✅ **Depth Map Reconstruction** - Recover 3D shape using Frankot-Chellappa or Poisson methods
- ✅ **Multi-Scene Support** - Support objects with different materials and shapes
- ✅ **High-Performance Processing** - Multi-threaded parallel computing for large-scale image processing

## 🏗️ Project Structure

```
images and code/
├── photometric stereo(for 's' datasets).py  # Main program
├── readme.txt                                # Dataset description
├── images_s1/                                # S1 scene images (Gypsum Hexahedron)
├── images_s2/                                # S2 scene images (Gypsum Cone)
├── images_s3/                                # S3 scene images (Foam Plastic Apple)
├── results_s1/                               # S1 processing results
├── results_s2/                               # S2 processing results
└── results_s3/                               # S3 processing results
```

## 📊 Dataset Description

### Scene Description

| Scene | Object Material | Shape |
|-------|----------------|-------|
| S1 | Gypsum | Hexahedron |
| S2 | Gypsum | Triangular Cone |
| S3 | Foamed Plastics | Apple |

### Light Source Configuration

Direction vectors of 5 light sources (normalized):

```
l1: [-35.3,  35.3, 50]  →  [-0.500,  0.500, 0.707]
l2: [  0.0,  50.0, 50]  →  [ 0.000,  0.707, 0.707]
l3: [ 50.0,   0.0, 50]  →  [ 0.707,  0.000, 0.707]
l4: [  0.0, -50.0, 50]  →  [ 0.000, -0.707, 0.707]
l5: [-35.3, -35.3, 50]  →  [-0.500, -0.500, 0.707]
```

### Camera Parameters

- **Lens**: 50mm
- **Aperture**: f/10
- **Position**: [-60, 0, 25] cm

## 🚀 Quick Start

### Requirements

```bash
numpy>=1.20.0
opencv-python>=4.5.0
scipy>=1.7.0
```

### Install Dependencies

```bash
pip install numpy opencv-python scipy
```

### Run the Program

Edit the `photometric stereo(for 's' datasets).py` file and modify the configuration parameters:

```python
SCENE_NAME = "s1"  # Select scene: "s1", "s2" or "s3"
SCALE_FACTOR = 0.25  # Image scaling factor (0-1)
```

Then run:

```python
python "photometric stereo(for 's' datasets).py"
```

## 📈 Processing Pipeline

### 1. Photometric Stereo

```
5 Input Images → Preprocessing → Light Matrix Solving → Normal Calculation → Albedo Extraction
```

**Step Details:**
- Image loading and cropping (remove 10% edge borders)
- Image scaling (default 25%)
- Background filtering (preserve object region)
- Multi-threaded parallel computation of normals and albedo

### 2. Depth Reconstruction

```
Normal Map → Gradient Calculation → Integration Reconstruction → Depth Normalization
```

**Supported Methods:**
- **Frankot-Chellappa Method**: Frequency domain global optimization, fast speed, high accuracy
- **Poisson Method**: Time domain iterative solution, adjustable iteration count

## 📁 Output Files

The following files are generated for each scene after processing:

### Visualization Images

| Filename | Description |
|----------|-------------|
| `normal_map_s{N}.png` | Normal map visualization |
| `albedo_map_s{N}.png` | Albedo map |
| `depth_map_s{N}_frankot_chellappa.png` | Depth map (Frankot-Chellappa method) |

### Raw Data

| Filename | Format | Description |
|----------|--------|-------------|
| `normal_map_float_s{N}.npy` | NumPy | Normal vector float data |
| `albedo_map_float_s{N}.npy` | NumPy | Albedo float data |
| `depth_raw_s{N}_frankot_chellappa.npy` | NumPy | Raw depth data |
| `mask_valid_s{N}.npy` | NumPy | Valid region mask |
| `light_matrix_s{N}.txt` | Text | Light direction matrix |

## 🔧 Technical Details

### Photometric Stereo Principle

For Lambertian surfaces, the brightness of each pixel satisfies:

```
I = ρ · (n · L)
```

Where:
- `I` - Observed intensity
- `ρ` - Albedo (surface reflectance)
- `n` - Unit surface normal vector
- `L` - Unit light direction vector

For 5 light sources, a system of linear equations can be established:

```
I₁ = ρ · (n · L₁)
I₂ = ρ · (n · L₂)
I₃ = ρ · (n · L₃)
I₄ = ρ · (n · L₄)
I₅ = ρ · (n · L₅)
```

The normal vector and albedo can be obtained through least squares solution.

### Depth Reconstruction Methods

#### Frankot-Chellappa Method

Solving Poisson equation in frequency domain:

```
∇²Z = ∇ · (-nₓ/nz, -nᵧ/nz)
```

Advantages: Global optimal solution, fast computation speed

#### Poisson Method

Iterative solution in time domain:

```
Z[i,j] = (Z[i-1,j] + Z[i+1,j] + Z[i,j-1] + Z[i,j+1] - div[i,j]) / 4
```

Advantages: High flexibility, controllable iteration precision

## 📊 Performance Optimization

- **Multi-threaded Processing**: Use `ThreadPoolExecutor` for parallel pixel computation
- **Block Processing**: Avoid memory overflow for large images
- **Progress Display**: Real-time processing progress feedback
- **Background Filtering**: Reduce invalid computations

## 🎯 Example Results

### S1 - Gypsum Hexahedron

**Normal Map**
<img src="images and code/results_s1/normal_map_s1.png" width="100%" />

**Albedo Map**
<img src="images and code/results_s1/albedo_map_s1.png" width="100%" />

**Depth Map**
<img src="images and code/results_s1/depth_map_s1_frankot_chellappa.png" width="100%" />

### S2 - Gypsum Cone

**Normal Map**
<img src="images and code/results_s2/normal_map_s2.png" width="100%" />

**Albedo Map**
<img src="images and code/results_s2/albedo_map_s2.png" width="100%" />

**Depth Map**
<img src="images and code/results_s2/depth_map_s2_frankot_chellappa.png" width="100%" />

### S3 - Foam Plastic Apple

**Normal Map**
<img src="images and code/results_s3/normal_map_s3.png" width="100%" />

**Albedo Map**
<img src="images and code/results_s3/albedo_map_s3.png" width="100%" />

**Depth Map**
<img src="images and code/results_s3/depth_map_s3_frankot_chellappa.png" width="100%" />

## 📝 Notes

1. **Image Requirements**: Ensure input images are named in the format `img1.jpg` to `img5.jpg`
2. **Memory Management**: Adjust `SCALE_FACTOR` appropriately when processing large images
3. **Background Filtering**: The program automatically filters background, but ensure clear contrast between object and background
4. **Light Direction**: Ensure vectors are correctly normalized when modifying the light matrix

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

This project is for learning and research purposes only.

## 🔗 References

- Woodham, R. J. (1980). "Photometric method for determining surface orientation from multiple images."
- Frankot, R. T., & Chellappa, R. (1988). "A method for enforcing integrability in shape from shading algorithms."

---

**Author**: JulianZhu  
**Email**: 1141911921@qq.com
