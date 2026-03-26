# Photometric Stereo Algorithm - Quantitative Evaluation and Test Sets
A complete framework for quantitative evaluation, test datasets, and visualization of photometric stereo algorithms based on the DiLiGenT-Pi dataset.

## Project Overview
This project implements a **quantitative evaluation framework for photometric stereo algorithms**, including standard experimental datasets, multi-dimensional evaluation metrics, visualization results, and automated analysis scripts. It enables quantitative evaluation and visual demonstration of 3D reconstruction results such as normals, depth maps, and albedo maps.

---

## Project Structure
```
Add code for quantitative evaluation, and provide test sets and test data_Bowen Liu
 ┣ Data analysis                     # Data analysis results (charts + reports)
 ┃ ┣ 01_Radar_Chart.png              # Radar chart: Multi-dimensional performance comparison
 ┃ ┣ 02_Score_Bar_Chart.png          # Score bar chart
 ┃ ┣ 03_Performance_Heatmap.png      # Performance heatmap
 ┃ ┣ 04_Box_Plots.png                # Box plot: Error distribution
 ┃ ┣ 05_3D_Scatter_Plot.png          # 3D scatter plot
 ┃ ┣ 06_Component_Scores_Radial.png  # Radial component score chart
 ┃ ┣ 07_Parallel_Coordinates.png     # Parallel coordinates plot
 ┃ ┣ 08_Grade_Distribution_Pie.png   # Pie chart: Grade distribution
 ┃ ┣ Conclusion.txt                  # Summary of evaluation conclusions
 ┃ ┣ evaluation_report.html          # Interactive HTML evaluation report
 ┃ ┗ evaluation_summary.csv          # Evaluation summary table
 ┣ Dataset for experiment(from DiLiGenT-Pi_release_png)  # Standard experimental dataset
 ┃ ┣ Astro / Bagua-R / Bagua-T / Bear / Bird / ...       # 10+ standard test objects
 ┃ ┃ ┣ 001-005.png                   # Input images under 5 different lighting conditions
 ┃ ┃ ┣ light_directions.txt          # Ground truth light direction annotations
 ┃ ┃ ┗ mask.png                      # Object mask image
 ┃ ┗ README.txt                      # Dataset documentation
 ┣ Evaluation                        # Text results of quantitative evaluation per object
 ┃ ┣ evaluation_XXX.txt              # Evaluation metrics for each test object
 ┣ some Result and Data              # Raw output data from algorithm reconstruction
 ┃ ┣ results_XXX                     # Reconstruction results categorized by object
 ┃ ┃ ┣ albedo_XXX.png                # Reconstructed albedo map
 ┃ ┃ ┣ depth_XXX.npy                 # Depth data (NumPy format)
 ┃ ┃ ┣ evaluation_XXX.json/txt       # Evaluation metrics files
 ┃ ┃ ┣ mask_XXX.npy                  # Mask data
 ┃ ┃ ┣ normal_XXX.png                # Reconstructed normal map
 ┃ ┃ ┗ pointcloud_XXX.ply            # 3D point cloud model
 ┣ The test of s1-s5                 # Custom test datasets (s1-s5)
 ┃ ┣ images_s1-s5                    # Custom test image sets
 ┃ ┣ results_s1-s2                   # Reconstruction results for custom test sets
 ┃ ┗ photometric_stereo_evalution.py # Evaluation script for custom test sets
 ┣ Visualization Results             # High-quality visualization results
 ┃ ┣ XXX folders                     # Categorized by object
 ┃ ┃ ┣ Albedo / Depth / Mask / Normal / Point Cloud visualizations
 ┣ Evaluation Metrics.txt            # Definition and explanation of evaluation metrics
 ┣ photometric_stereo_evalution_for_dataset.py  # Main evaluation script for standard dataset
```

---

## Core Features
1. **Photometric Stereo Reconstruction**
   Input: Multi-lighting images → Output: Normal maps, depth maps, albedo maps, 3D point clouds
2. **Automated Quantitative Evaluation**
   Supports both standard and custom datasets, automatically calculating error metrics and generating reports
3. **Comprehensive Visual Analysis**
   8 professional data analysis charts + reconstruction result visualizations for intuitive performance demonstration
4. **Standardized Test Sets**
   Based on the official DiLiGenT-Pi dataset, including 10+ common test objects with annotations

---

## File Description
### 1. Core Scripts
| File Name | Function |
|-----------|----------|
| `photometric_stereo_evalution_for_dataset.py` | Main batch evaluation script for the standard dataset |
| `photometric_stereo_evalution.py` | Evaluation script for custom test sets (s1-s5) |

### 2. Data Folders
- `Dataset for experiment`: Official standard test set with images, lighting annotations, and masks
- `some Result and Data`: Raw reconstruction output for secondary analysis
- `Visualization Results`: High-definition results for papers and reports

### 3. Evaluation Outputs
- `Evaluation`: Plain-text evaluation metrics
- `Data analysis`: Visualization charts, CSV summaries, and interactive HTML reports

---

## Usage
### (1) Standard Dataset Evaluation
```bash
python photometric_stereo_evalution_for_dataset.py
```
You can **modify the target data folder path in the code** to specify the object you want to evaluate. After running, the script will automatically generate:
- `Bagua-R Albedo Map.png`
- `Depth Map.png`
- `Input.png`
- `Mask.png`
- `Normal Components.png`
- `Normal Map(RGB).png`
- `Point Cloud.png`
- Corresponding evaluation files

### (2) Custom Test Set Evaluation
```bash
cd "The test of s1-s5"
python photometric_stereo_evalution.py
```

### ⚠️ Important Notice for Custom Dataset
**If you use a custom dataset for computation, you must manually modify the light path matrix, camera intrinsic parameters, camera extrinsic parameters, and camera position in the code before running the script.**

---

## Evaluation Metrics
Core quantitative evaluation metrics (see `Evaluation Metrics.txt`):
- Normal estimation error (angular error, mean/median/std)
- Depth reconstruction accuracy
- Albedo reconstruction error
- Overall reconstruction quality score

---

## Dependencies
- Python 3.x
- NumPy / OpenCV / Pillow
- Matplotlib / Seaborn (data visualization)
- Open3D / Plyfile (point cloud processing)

---

## Contributors
- **Chengting Sheng**: Provided custom test datasets; optimized code for point cloud reconstruction and depth map generation.
- **Bowen Liu**: Photometric stereo algorithm research; developed quantitative evaluation framework; conducted dataset testing and data summary & analysis.

---

## License
This project is released under the **MIT License**.

---

## Author
**Bowen Liu**