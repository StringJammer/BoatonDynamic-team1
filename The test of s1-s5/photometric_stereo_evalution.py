# photometric_stereo_for_real_scenes.py
"""
Improved photometric stereo 3D reconstruction code (stable version)
- Mask generation: based on maximum brightness across multiple images + quantile threshold + morphology (from user code)
- Depth reconstruction: perspective projection model, sparse linear system with correct coefficients (solved with LSQR)
- Point cloud generation: camera coordinates to world coordinates, optional statistical filtering
- All filtering options are off by default; parameters can be adjusted as needed
- Added simplified quantitative evaluation with stricter criteria
"""

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import os
import glob
from scipy.ndimage import binary_opening, binary_closing
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial import KDTree
import json

# ==================== User-adjustable parameters ====================

Scenes = ['s2']                         # Select images to process
target_width = 1000                     # Target image width after resizing (height scaled accordingly)

# Mask generation
mask_percentile = 0.15                  # Percentage of brightest pixels to retain (0~1); adjust based on object size: s1:0.25, s2:0.15, s3:0.05
use_morphology = True                   # Whether to use morphological closing to fill holes

# Point cloud statistical filtering (optional, disabled by default)
use_statistical_outlier_removal = False
stat_nb_neighbors = 100
stat_std_ratio = 0.2

# Evaluation parameters
enable_evaluation = True                # Enable comprehensive evaluation
save_evaluation_report = True           # Save detailed evaluation report to TXT file

# World scale (if absolute size is unknown, keep as 1.0)
pixel_to_world = 0.001                  # World units per pixel (e.g., meters/pixel)
# =====================================================

# -------------------- Light directions (consistent with simulation) --------------------
light_positions = np.array([
    [0.353, 0.353, -0.5],
    [0, 0.5, -0.5],
    [-0.5, 0, -0.5],
    [0, -0.5, -0.5],
    [0.353, -0.353, -0.5]
], dtype=np.float32)
light_directions = light_positions / np.linalg.norm(light_positions, axis=1, keepdims=True)
print("Light directions:\n", light_directions)

# -------------------- Camera parameters (consistent with simulation) --------------------
camera_pos = [0.6, 0, -0.25]                     # Camera position
C = np.array(camera_pos, dtype=np.float32)
look_at = np.array([0, 0, 0])
forward = look_at - C
forward = forward / np.linalg.norm(forward)
world_up = np.array([0, 0, 1])                   # World up direction
right = np.cross(forward, world_up)
right = right / np.linalg.norm(right)
up = np.cross(right, forward)                    # Camera up direction
up = up / np.linalg.norm(up)

# Camera intrinsics (full-frame 50mm lens, original size 7360x4912)
orig_width, orig_height = 7360, 4912
sensor_width_mm = 36.0
focal_length_mm = 50.0
f_pixel_orig = focal_length_mm / sensor_width_mm * orig_width
cx_orig = orig_width / 2.0
cy_orig = orig_height / 2.0

# After scaling
scale = target_width / orig_width
target_height = int(orig_height * scale)
f = f_pixel_orig * scale
cx = target_width / 2.0
cy = target_height / 2.0

print(f"Resized image size: {target_width} x {target_height}")
print(f"Focal length (pixels): {f:.2f}, principal point: ({cx:.2f}, {cy:.2f})")

# ==================== Utility functions ====================

def compute_mask_from_images(images, percentile=0.7, morph=True):
    """
    Generate mask based on maximum brightness across multiple images (adapted from user code).
    images: list of 2D arrays (H,W) with values in [0,1]
    percentile: percentage of brightest pixels to retain (0~1)
    morph: whether to apply morphological opening and closing
    """
    max_img = np.max(images, axis=0)
    nonzero = max_img[max_img > 0]
    if len(nonzero) == 0:
        return np.zeros_like(max_img, dtype=bool)
    thresh = np.percentile(nonzero, 100 * (1 - percentile))   # keep top (percentile)% pixels
    mask = max_img > thresh
    if morph:
        structure = np.ones((5, 5))
        mask = binary_opening(mask, structure=structure)
        mask = binary_closing(mask, structure=structure)
    return mask

def photometric_stereo_parallel(images, light_dirs, mask_init):
    """
    Standard photometric stereo (directional light).
    images: list of 2D arrays (H,W) with values in [0,1]
    light_dirs: (5,3) light directions (unit vectors)
    mask_init: initial mask
    Returns: normal_map (H,W,3), albedo_map (H,W), mask (H,W)
    """
    h, w = images[0].shape
    imgs_stack = np.stack(images, axis=0)           # (5, H, W)
    I_mat = imgs_stack.reshape(5, -1)               # (5, N)
    L_pinv = np.linalg.pinv(light_dirs)             # (3,5)
    G = L_pinv @ I_mat                               # (3, N)
    rho = np.linalg.norm(G, axis=0)                  # (N,)
    mask_rho = rho > 1e-4
    mask_valid = mask_rho & mask_init.ravel()
    normal = np.zeros_like(G.T)                       # (N,3)
    normal[mask_valid] = (G[:, mask_valid] / rho[mask_valid]).T
    normal_map = normal.reshape(h, w, 3)
    albedo_map = rho.reshape(h, w)
    mask_final = mask_valid.reshape(h, w)
    return normal_map, albedo_map, mask_final

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
            # Correct constraint: (-A + n_x/f) * z_i,j + A * z_i,j+1 = 0
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

# ==================== MODIFIED: Stricter Evaluation Function ====================
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
    scenes = Scenes
    for scene in scenes:
        print("\n" + "="*60)
        print(f"Processing scene: {scene}")
        print("="*60)

        result_dir = f"results_{scene}"
        image_dir = f"images_{scene}"
        os.makedirs(result_dir, exist_ok=True)

        if not os.path.isdir(image_dir):
            print(f"Directory {image_dir} does not exist, skipping")
            continue

        img_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))[:5]
        if len(img_files) != 5:
            print(f"Need 5 images, found {len(img_files)}, skipping")
            continue

        # Show the first raw image for verification
        sample = cv2.imread(img_files[0])
        sample_rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6,6))
        plt.imshow(sample_rgb)
        plt.title(f"Sample Input - {scene}")
        plt.axis('off')
        plt.show()

        # Read and preprocess images
        imgs_norm = []
        for fname in img_files:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img_resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
            img_norm = img_resized / 255.0
            imgs_norm.append(img_norm)

        H, W = imgs_norm[0].shape
        print(f"Image size: {H} x {W}")

        # Generate initial mask
        mask_init = compute_mask_from_images(imgs_norm, percentile=mask_percentile, morph=use_morphology)
        plt.figure(figsize=(6,6))
        plt.imshow(mask_init, cmap='gray')
        plt.title(f"Initial Mask - {scene}")
        plt.axis('off')
        plt.show()

        # Photometric stereo normal estimation
        print("Running photometric stereo...")
        normal_map, albedo_map, mask = photometric_stereo_parallel(imgs_norm, light_directions, mask_init)

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

        # Transform normals to camera coordinate system
        n_cam = np.zeros_like(normal_map)
        n_cam[..., 0] = np.sum(normal_map * right, axis=2)
        n_cam[..., 1] = np.sum(normal_map * up, axis=2)
        n_cam[..., 2] = np.sum(normal_map * forward, axis=2)

        # Prepare data for depth reconstruction
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
            continue

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

        # Save intermediate results
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

        # Generate point cloud
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

        print(f"Scene {scene} finished.\n")

    print("All scenes processed!")

if __name__ == "__main__":
    main()