# photometric_stereo_evalution_for_dataset/py
"""
Photometric stereo 3D reconstruction for custom datasets
- Supports PNG format images
- Uses provided mask.png file
- Loads light directions from light_directions.txt file
- Stricter evaluation criteria
- Complete reconstruction and evaluation pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import os
import glob
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial import KDTree
import json

# ==================== User-adjustable parameters ====================

# Dataset configuration
dataset_folder = "Astro"  
image_prefix = "00"      # Image prefix: "00" for 001.png, 002.png, etc.
image_format = "png"     # Image format: "png"


target_width = 1216  
target_height = 1216

# Point cloud statistical filtering (optional, disabled by default)
use_statistical_outlier_removal = False
stat_nb_neighbors = 100
stat_std_ratio = 0.2

# Evaluation parameters
enable_evaluation = True                # Enable comprehensive evaluation
save_evaluation_report = True           # Save detailed evaluation report to TXT file

# World scale (adjusted for 1.5m distance and 5cm object)
pixel_to_world = 0.0123                 # World units per pixel (meters/pixel)
# =====================================================

# -------------------- Light directions --------------------
light_positions = np.array([
    [0.353, 0.353, -0.5],
    [0, 0.5, -0.5],
    [-0.5, 0, -0.5],
    [0, -0.5, -0.5],
    [0.353, -0.353, -0.5]
], dtype=np.float32)
light_directions = light_positions / np.linalg.norm(light_positions, axis=1, keepdims=True)
print("Default light directions (if file not found):")
for i, direction in enumerate(light_directions):
    print(f"  Light {i+1}: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")

# -------------------- Camera parameters for YOUR setup --------------------
camera_pos = [0, 0, 1.5]                     # Camera position: 1.5m above object
C = np.array(camera_pos, dtype=np.float32)
look_at = np.array([0, 0, 0])                # Look at object center
forward = look_at - C
forward = forward / np.linalg.norm(forward)
world_up = np.array([0, 1, 0])               # World up direction: Y-axis
right = np.cross(forward, world_up)
right = right / np.linalg.norm(right)
up = np.cross(right, forward)                # Camera up direction
up = up / np.linalg.norm(up)

# Camera intrinsics for MER-503-36U3C with 50mm lens
focal_length_mm = 50.0                       # 50mm lens
sensor_width_mm = 8.8                        # 2/3 inch sensor width
image_width_crop = 1216                      # Your cropped image width
image_height_crop = 1216                     # Your cropped image height

# Calculate focal length in pixels
# 公式: f_pixels = (focal_length_mm / sensor_width_mm) * image_width_crop
sensor_width_crop_mm = sensor_width_mm * (image_width_crop / 2448)
f = (focal_length_mm / sensor_width_crop_mm) * image_width_crop

# Principal point (center of image)
cx = target_width / 2.0
cy = target_height / 2.0

print("="*60)
print("CAMERA PARAMETERS FOR YOUR SETUP")
print("="*60)
print(f"Camera: MER-503-36U3C with 50mm lens")
print(f"Image resolution: {target_width} x {target_height}")
print(f"Focal length (pixels): {f:.2f}")
print(f"Principal point: ({cx:.2f}, {cy:.2f})")
print(f"Object distance: 1.5m")
print(f"World scale: {pixel_to_world} m/pixel")
print("="*60)

# ==================== Utility functions ====================

def load_light_directions_from_file(filepath):
    """Load light directions from file, fallback to defaults if not found"""
    if os.path.exists(filepath):
        print(f"Loading light directions from {filepath}")
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            vectors = []
            for line in lines:
                if '#' in line:
                    line = line.split('#')[0]
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = map(float, parts[:3])
                    vectors.append([x, y, z])
            
            if len(vectors) >= 5:
                vectors = np.array(vectors[:5], dtype=np.float32)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                light_dirs = vectors / norms
                
                print("Light directions from file:")
                for i, direction in enumerate(light_dirs):
                    print(f"  Light {i+1}: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
                
                return light_dirs
            else:
                print(f"Warning: Only {len(vectors)} directions found, using defaults")
        except Exception as e:
            print(f"Error loading light directions: {e}, using defaults")
    
    print("Using default light directions")
    return light_directions

def load_provided_mask(mask_path, target_size):
    """
    Load and preprocess the provided mask file.
    
    Args:
        mask_path: Path to the mask file (e.g., mask.png)
        target_size: Target size (width, height) for the mask
        
    Returns:
        mask: Boolean mask array
    """
    # Load mask image
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Resize mask to target size
    mask_resized = cv2.resize(mask_img, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Convert to boolean (assuming mask is binary: 0=background, >0=object)
    mask = mask_resized > 0
    
    # Ensure mask is boolean
    mask = mask.astype(bool)
    
    print(f"Loaded mask from {mask_path}")
    print(f"Mask size: {mask.shape}, object pixels: {np.sum(mask)}")
    
    return mask

def photometric_stereo_parallel(images, light_dirs, mask):
    """
    Standard photometric stereo (directional light).
    images: list of 2D arrays (H,W) with values in [0,1]
    light_dirs: (5,3) light directions (unit vectors)
    mask: boolean mask (H,W) - provided by user
    Returns: normal_map (H,W,3), albedo_map (H,W)
    """
    h, w = images[0].shape
    imgs_stack = np.stack(images, axis=0)           # (5, H, W)
    I_mat = imgs_stack.reshape(5, -1)               # (5, N)
    L_pinv = np.linalg.pinv(light_dirs)             # (3,5)
    G = L_pinv @ I_mat                               # (3, N)
    rho = np.linalg.norm(G, axis=0)                  # (N,)
    mask_rho = rho > 1e-4
    mask_valid = mask_rho & mask.ravel()
    normal = np.zeros_like(G.T)                       # (N,3)
    normal[mask_valid] = (G[:, mask_valid] / rho[mask_valid]).T
    normal_map = normal.reshape(h, w, 3)
    albedo_map = rho.reshape(h, w)
    return normal_map, albedo_map

def build_depth_system(n_cam, x_prime, y_prime, mask, cx, cy, f):
    """
    Build sparse linear system A * z = b (correct coefficient version).
    n_cam: (H,W,3) normals in camera coordinates (components: right, up, forward)
    x_prime, y_prime: normalized image coordinates (u-cx)/f, (v-cy)/f
    mask: valid pixel mask
    Returns: A (csr_matrix), b (np.array), valid_i, valid_j, idx_map
    """
    H, W = mask.shape
    valid_i, valid_j = np.where(mask)
    N_valid = len(valid_i)
    if N_valid == 0:
        return None, None, None, None, None

    # Create mapping from coordinates to linear index
    idx_map = -np.ones((H, W), dtype=int)
    idx_map[valid_i, valid_j] = np.arange(N_valid)

    rows, cols, data = [], [], []
    b = []
    eq_idx = 0

    for idx in range(N_valid):
        i, j = valid_i[idx], valid_j[idx]
        n_x = n_cam[i, j, 0]
        n_y = n_cam[i, j, 1]
        n_z = n_cam[i, j, 2]
        xp = x_prime[i, j]
        yp = y_prime[i, j]
        A_val = n_x * xp + n_y * yp + n_z   # A in the formula

        # Horizontal neighbor (i, j+1)
        if j+1 < W and mask[i, j+1]:
            nbr = idx_map[i, j+1]
            rows.extend([eq_idx, eq_idx])
            cols.extend([idx, nbr])
            data.extend([-A_val + n_x / f, A_val])
            b.append(0.0)
            eq_idx += 1

        # Vertical neighbor (i+1, j)
        if i+1 < H and mask[i+1, j]:
            nbr = idx_map[i+1, j]
            rows.extend([eq_idx, eq_idx])
            cols.extend([idx, nbr])
            data.extend([-A_val + n_y / f, A_val])
            b.append(0.0)
            eq_idx += 1

    # Seed point constraint (take a point near the center of the valid region)
    if N_valid > 0:
        # Compute center point
        center_i = int(np.mean(valid_i))
        center_j = int(np.mean(valid_j))
        # Ensure the center point lies within the mask
        if not mask[center_i, center_j]:
            center_i, center_j = valid_i[N_valid//2], valid_j[N_valid//2]
        seed_idx = idx_map[center_i, center_j]
        rows.append(eq_idx)
        cols.append(seed_idx)
        data.append(1.0)
        b.append(1.0)       # Set seed point depth to 1.0 (relative scale)
        eq_idx += 1

    A = csr_matrix((data, (rows, cols)), shape=(eq_idx, N_valid))
    b = np.array(b)
    return A, b, valid_i, valid_j, idx_map

# ==================== Evaluation Function ====================
def evaluate_reconstruction_strict(scene_name, depth_map, normal_map, mask, point_cloud, result_dir):
    """
    Reconstruction quality evaluation with strict criteria
    Returns key metrics suitable for cross-scene comparison
    """
    print("\n" + "="*60)
    print("Reconstruction Quality Evaluation")
    print("="*60)
    
    # Get point cloud data
    points_array = np.asarray(point_cloud.points)
    num_points = len(points_array)
    
    # 1. Geometric completeness metrics
    valid_depth = depth_map[mask]
    valid_pixels = np.sum(~np.isnan(valid_depth))
    total_pixels = np.sum(mask)
    completeness_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0
    
    depth_values = valid_depth[~np.isnan(valid_depth)]
    if len(depth_values) > 0:
        depth_mean = float(np.mean(depth_values))
        depth_std = float(np.std(depth_values))
    else:
        depth_mean = 0
        depth_std = 0
    
    # 2. Normal vector quality
    normals = normal_map[mask]
    if len(normals) > 0:
        magnitudes = np.linalg.norm(normals, axis=1)
        valid_normals = np.sum((magnitudes >= 0.9) & (magnitudes <= 1.1))
        valid_normals_ratio = valid_normals / len(magnitudes)
    else:
        valid_normals_ratio = 0
    
    # 3. Point cloud quality
    if num_points > 0:
        if num_points > 1:
            kdtree = KDTree(points_array)
            distances, _ = kdtree.query(points_array, k=2)
            nearest_distances = distances[:, 1]
            uniformity_index = float(np.std(nearest_distances) / (np.mean(nearest_distances) + 1e-8))
        else:
            uniformity_index = 0
    else:
        uniformity_index = 0
    
    # 4. Algorithm stability
    if len(depth_values) > 0:
        laplacian = cv2.Laplacian(depth_map, cv2.CV_64F)
        laplacian_masked = laplacian[mask]
        laplacian_mean = float(np.nanmean(np.abs(laplacian_masked)))
    else:
        laplacian_mean = 0
    
    # Calculate component scores with strict criteria
    component_scores = {}
    
    # 1. Completeness score (0-25 points)
    component_scores['completeness'] = min(25.0, completeness_ratio * 25)
    
    # 2. Normal quality score (0-25 points)
    component_scores['normal_quality'] = min(25.0, valid_normals_ratio * 25)
    
    # 3. Point cloud score (0-25 points) - strict formula
    if num_points > 0:
        base_count_score = np.tanh(num_points / 50000) * 12
        uniformity_adjustment = (0.3 - uniformity_index) * 20
        point_cloud_score = base_count_score + uniformity_adjustment
        component_scores['point_cloud'] = float(np.clip(point_cloud_score, 0, 25))
    else:
        component_scores['point_cloud'] = 0.0
    
    # 4. Smoothness score (0-25 points) - strict formula
    component_scores['smoothness'] = float(max(0.0, 25.0 - laplacian_mean * 3000))
    
    # Total score
    total_score = sum(component_scores.values())
    
    # Determine grade
    if total_score >= 90:
        grade = "Excellent"
    elif total_score >= 80:
        grade = "Good"
    elif total_score >= 70:
        grade = "Fair"
    elif total_score >= 60:
        grade = "Pass"
    else:
        grade = "Poor"
    
    # Organize results
    evaluation_results = {
        'scene_name': scene_name,
        'key_metrics': {
            'valid_pixel_ratio': float(completeness_ratio),
            'num_points': int(num_points),
            'valid_normals_ratio': float(valid_normals_ratio),
            'depth_mean': depth_mean,
            'depth_std': depth_std,
            'uniformity_index': uniformity_index,
            'laplacian_mean': laplacian_mean
        },
        'component_scores': component_scores,
        'total_score': float(total_score),
        'grade': grade
    }
    
    # Save JSON report
    json_report_path = os.path.join(result_dir, f"evaluation_{scene_name}.json")
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    # Generate TXT report
    txt_report_path = os.path.join(result_dir, f"evaluation_{scene_name}.txt")
    generate_strict_txt_report(evaluation_results, txt_report_path)
    
    # Print summary
    print("Key Evaluation Metrics")
    print("-"*40)
    print(f"Scene: {scene_name}")
    print(f"Valid Pixel Ratio: {completeness_ratio*100:.1f}%")
    print(f"Valid Normals Ratio: {valid_normals_ratio*100:.1f}%")
    print(f"Number of Points: {num_points:,}")
    print(f"Depth Smoothness: {laplacian_mean:.6f}")
    print(f"Uniformity Index: {uniformity_index:.3f}")
    print(f"Total Score: {total_score:.1f}/100")
    print(f"Grade: {grade}")
    
    print(f"\nReports saved to {result_dir}:")
    print(f"TXT Report: evaluation_{scene_name}.txt")
    print(f"JSON Data: evaluation_{scene_name}.json")
    print("="*60)
    
    return evaluation_results

def generate_strict_txt_report(evaluation_results, filepath):
    """
    Generate strict TXT format evaluation report
    """
    scene_name = evaluation_results['scene_name']
    key_metrics = evaluation_results['key_metrics']
    component_scores = evaluation_results['component_scores']
    total_score = evaluation_results['total_score']
    grade = evaluation_results['grade']
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Photometric Stereo 3D Reconstruction Evaluation Report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Scene: {scene_name}\n\n")
        
        f.write("1. OVERALL RESULTS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total Score: {total_score:.1f}/100\n")
        f.write(f"Grade: {grade}\n\n")
        
        f.write("2. KEY METRICS\n")
        f.write("-"*40 + "\n")
        
        f.write("2.1 Geometric Completeness\n")
        f.write(f"Valid Pixel Ratio: {key_metrics['valid_pixel_ratio']*100:.2f}%\n")
        f.write(f"Mean Depth: {key_metrics['depth_mean']:.6f}\n")
        f.write(f"Depth Std Dev: {key_metrics['depth_std']:.6f}\n\n")
        
        f.write("2.2 Normal Vector Quality\n")
        f.write(f"Valid Normals Ratio: {key_metrics['valid_normals_ratio']*100:.2f}%\n\n")
        
        f.write("2.3 Point Cloud Quality\n")
        f.write(f"Number of Points: {key_metrics['num_points']:,}\n")
        f.write(f"Uniformity Index: {key_metrics['uniformity_index']:.6f} (smaller is better)\n\n")
        
        f.write("2.4 Algorithm Stability\n")
        f.write(f"Depth Smoothness: {key_metrics['laplacian_mean']:.6f} (smaller is better)\n\n")
        
        f.write("3. COMPONENT SCORES (0-25 points each)\n")
        f.write("-"*40 + "\n")
        f.write(f"Completeness Score: {component_scores['completeness']:.1f}\n")
        f.write(f"Normal Quality Score: {component_scores['normal_quality']:.1f}\n")
        f.write(f"Point Cloud Score: {component_scores['point_cloud']:.1f}\n")
        f.write(f"Smoothness Score: {component_scores['smoothness']:.1f}\n\n")
        
        f.write("4. RECOMMENDATIONS\n")
        f.write("-"*40 + "\n")
        
        if total_score >= 90:
            f.write("Reconstruction quality is excellent. The result is suitable for high-precision applications.\n")
        elif total_score >= 80:
            f.write("Reconstruction quality is good. Suitable for most applications.\n")
            if key_metrics['valid_pixel_ratio'] < 0.85:
                f.write("Consider optimizing mask parameters to improve coverage.\n")
            if key_metrics['uniformity_index'] > 0.4:
                f.write("Point distribution could be more uniform.\n")
        elif total_score >= 70:
            f.write("Reconstruction quality is fair. Room for improvement.\n")
            f.write("Optimize lighting conditions and camera parameters.\n")
        elif total_score >= 60:
            f.write("Reconstruction quality is minimally acceptable. Significant improvements needed.\n")
            f.write("Consider redesigning experimental setup or using more robust methods.\n")
        else:
            f.write("Reconstruction quality is insufficient. Major improvements required.\n")
            f.write("Check hardware, input image quality, and algorithm parameters.\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("End of Report\n")
        f.write("="*60 + "\n")

# ==================== Main pipeline ====================

def main():
    # Use the dataset folder as scene name
    scene = dataset_folder
    print("\n" + "="*60)
    print(f"Processing dataset: {scene}")
    print("="*60)

    result_dir = f"results_{scene}"
    image_dir = dataset_folder
    os.makedirs(result_dir, exist_ok=True)

    if not os.path.isdir(image_dir):
        print(f"Directory {image_dir} does not exist, skipping")
        return

    # Load light directions from file
    light_file = os.path.join(image_dir, "light_directions.txt")
    light_dirs = load_light_directions_from_file(light_file)
    
    if light_dirs is None or len(light_dirs) != 5:
        print("ERROR: Could not load proper light directions")
        return

    # Find all PNG images with the specified naming pattern
    img_files = sorted(glob.glob(os.path.join(image_dir, f"{image_prefix}*.{image_format}")))[:5]
    
    if len(img_files) < 5:
        print(f"Found {len(img_files)} images, but need 5 images. Looking for alternative naming patterns...")
        # Try alternative pattern
        img_files = sorted(glob.glob(os.path.join(image_dir, f"*.{image_format}")))[:5]
    
    if len(img_files) != 5:
        print(f"Need 5 images, found {len(img_files)}, skipping")
        return

    print(f"Found images: {[os.path.basename(f) for f in img_files]}")

    # Show the first raw image for verification
    sample = cv2.imread(img_files[0])
    sample_rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6,6))
    plt.imshow(sample_rgb)
    plt.title(f"Sample Input - {scene}")
    plt.axis('off')
    plt.show()

    # ---------- Read and preprocess images ----------
    imgs_norm = []
    for fname in img_files:
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        # Check if image needs resizing
        if img.shape[1] != target_width or img.shape[0] != target_height:
            print(f"Resizing image from {img.shape[1]}x{img.shape[0]} to {target_width}x{target_height}")
            img_resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img
            
        img_norm = img_resized / 255.0
        imgs_norm.append(img_norm)

    H, W = imgs_norm[0].shape
    print(f"Image size: {H} x {W}")

    # ---------- Load provided mask ----------
    mask_path = os.path.join(image_dir, "mask.png")
    if not os.path.exists(mask_path):
        # Try alternative mask names
        mask_alternatives = ["mask.png", "Mask.png", "MASK.png", "mask.jpg", "mask.jpeg"]
        for mask_name in mask_alternatives:
            mask_path = os.path.join(image_dir, mask_name)
            if os.path.exists(mask_path):
                break
    
    if os.path.exists(mask_path):
        mask = load_provided_mask(mask_path, (W, H))
        
        # Display the mask
        plt.figure(figsize=(6,6))
        plt.imshow(mask, cmap='gray')
        plt.title(f"Provided Mask - {scene}")
        plt.axis('off')
        plt.show()
    else:
        print(f"Mask file not found in {image_dir}. Please ensure mask.png is in the dataset folder.")
        return

    # ---------- Photometric stereo normal estimation ----------
    print("Running photometric stereo...")
    normal_map, albedo_map = photometric_stereo_parallel(imgs_norm, light_dirs, mask)

    # Re-normalize normals
    norm_n = np.linalg.norm(normal_map, axis=-1, keepdims=True)
    normal_map = normal_map / (norm_n + 1e-8)

    # Display normal map
    normal_vis = (normal_map + 1.0) / 2.0
    plt.figure(figsize=(8,8))
    plt.imshow(normal_vis)
    plt.title(f"Normal Map (RGB) - {scene}")
    plt.axis('off')
    plt.show()

    # Display normal components
    fig, axes = plt.subplots(1,3, figsize=(15,5))
    for k, title in enumerate(['X (right)', 'Y (up)', 'Z (forward)']):
        im = axes[k].imshow(normal_map[..., k], cmap='coolwarm', vmin=-1, vmax=1)
        axes[k].set_title(title)
        axes[k].axis('off')
        plt.colorbar(im, ax=axes[k])
    plt.suptitle(f"Normal Components - {scene}")
    plt.tight_layout()
    plt.show()

    # Display albedo map
    plt.figure(figsize=(6,6))
    plt.imshow(albedo_map, cmap='gray')
    plt.title(f"Albedo Map - {scene}")
    plt.axis('off')
    plt.show()

    # ---------- Transform normals to camera coordinate system ----------
    n_cam = np.zeros_like(normal_map)
    n_cam[..., 0] = np.sum(normal_map * right, axis=2)
    n_cam[..., 1] = np.sum(normal_map * up, axis=2)
    n_cam[..., 2] = np.sum(normal_map * forward, axis=2)

    # ---------- Prepare data for depth reconstruction ----------
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(u, v)
    x_prime = (u - cx) / f
    y_prime = (v - cy) / f

    # Build linear system
    print("Building depth solving linear system...")
    A, b, valid_i, valid_j, idx_map = build_depth_system(n_cam, x_prime, y_prime, mask, cx, cy, f)
    if A is None:
        print("Warning: no valid pixels, skipping this scene")
        return

    # Solve for depth
    print("Solving for depth...")
    z_sol, istop, itn, r1norm = lsqr(A, b, atol=1e-6, btol=1e-6)[:4]
    print(f"LSQR finished, iterations {itn}, residual {r1norm}")

    # Depth map
    depth_map = np.zeros((H, W))
    depth_map[valid_i, valid_j] = z_sol
    depth_map[~mask] = np.nan

    # Display depth map
    valid_depth = depth_map[mask]
    if len(valid_depth) > 0:
        d_min, d_max = np.percentile(valid_depth, [1, 99])
        depth_disp = np.zeros_like(depth_map)
        depth_disp[mask] = (valid_depth - d_min) / (d_max - d_min + 1e-8)
        plt.figure(figsize=(6,6))
        plt.imshow(depth_disp, cmap='viridis')
        plt.colorbar()
        plt.title(f"Depth Map - {scene}")
        plt.axis('off')
        plt.show()

    # ---------- Save intermediate results ----------
    cv2.imwrite(os.path.join(result_dir, f"normal_{scene}.png"),
                (normal_vis * 255).clip(0,255).astype(np.uint8))
    albedo_norm = albedo_map.copy()
    if mask.any():
        a_min, a_max = albedo_norm[mask].min(), albedo_norm[mask].max()
        albedo_norm = (albedo_norm - a_min) / (a_max - a_min + 1e-8)
    albedo_norm[~mask] = 0
    cv2.imwrite(os.path.join(result_dir, f"albedo_{scene}.png"),
                (albedo_norm * 255).astype(np.uint8))
    np.save(os.path.join(result_dir, f"depth_{scene}.npy"), depth_map)
    np.save(os.path.join(result_dir, f"mask_{scene}.npy"), mask)

    # ---------- Generate point cloud ----------
    print("Generating point cloud...")
    points, colors = [], []
    for idx in range(len(valid_i)):
        i, j = valid_i[idx], valid_j[idx]
        z = depth_map[i, j]
        # Camera coordinates
        Xc = z * x_prime[i, j]
        Yc = z * y_prime[i, j]
        Zc = z
        # Transform to world coordinates
        P_world = C + Xc * right + Yc * up + Zc * forward
        points.append(P_world)
        # Use normals as color
        n = normal_map[i, j]
        colors.append((n + 1) / 2)

    points = np.array(points)
    colors = np.array(colors)

    # Scale to physical units
    points = points * pixel_to_world

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Optional: statistical filtering
    if use_statistical_outlier_removal:
        print("Removing outliers with statistical filtering...")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=stat_nb_neighbors,
                                                   std_ratio=stat_std_ratio)
        print(f"Points after filtering: {len(pcd.points)}")

    # Save point cloud
    ply_path = os.path.join(result_dir, f"pointcloud_{scene}.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Point cloud saved to: {ply_path}")

    # Run strict evaluation
    if enable_evaluation:
        evaluation_results = evaluate_reconstruction_strict(
            scene_name=scene,
            depth_map=depth_map,
            normal_map=normal_map,
            mask=mask,
            point_cloud=pcd,
            result_dir=result_dir
        )

    # Visualize point cloud
    print("Opening point cloud visualization window...")
    o3d.visualization.draw_geometries([pcd], window_name=f"Scene {scene}")

    print(f"Dataset {scene} processing finished.\n")
    print("All processing completed!")

if __name__ == "__main__":
    # Before running, set your dataset folder name
    print("IMPORTANT: Update the 'dataset_folder' variable at the top of the script")
    print(f"Current dataset folder: {dataset_folder}")
    print("\nTo use this code with your dataset:")
    print("1. Update 'dataset_folder' to your folder name")
    print("2. Ensure folder contains: 001.png-005.png, mask.png, light_directions.txt")
    print("3. Images should be 1216x1216 pixels")
    print("4. Run the script\n")
    
    # Ask for confirmation
    response = input(f"Do you want to proceed with dataset folder '{dataset_folder}'? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        main()
    else:
        print("Please update the dataset_folder variable and run the script again.")