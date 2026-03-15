import numpy as np
import cv2
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from scipy.fft import fft2, ifft2, fftfreq

# ==================== Part 1: Photometric Stereo (5 lights) ====================

def process_pixel(y, x, imgs_2d, L_pinv):
    """Multi-threaded pixel processing"""
    I = imgs_2d[:, y, x]
    if np.all(I == 0.0):
        return (y, x, np.zeros(3), 0.0)
    
    g = L_pinv @ I
    rho = np.linalg.norm(g)
    
    if rho > 1e-4:
        normal = g / rho
    else:
        normal = np.zeros(3)
        rho = 0.0
    return (y, x, normal, rho)

def photometric_stereo_scene(image_dir: str, save_dir: str, scale_factor: float = 0.25, scene_name: str = "s1"):
    """
    Photometric stereo main function (5 lights version)
    
    Parameters:
    image_dir: Image folder path
    save_dir: Result save path
    scale_factor: Image scaling factor (0-1), default 0.25 (reduce to 25%)
    scene_name: Scene name (s1, s2, s3)
    """
    # 5 light source direction matrix
    L = np.array([
        [-0.500,  0.500,  0.707],  # l1: top-right
        [ 0.000,  0.707,  0.707],  # l2: top
        [ 0.707,  0.000,  0.707],  # l3: right
        [ 0.000, -0.707,  0.707],  # l4: bottom
        [-0.500, -0.500,  0.707]   # l5: bottom-left
    ], dtype=np.float32)

    os.makedirs(save_dir, exist_ok=True)
    print(f"================ Starting Photometric Stereo Processing ================")
    print(f"Scene: {scene_name}")
    print(f"Image directory: {image_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Scale factor: {scale_factor}")
    print(f"Light sources: 5")

    # Load images - match 5 images
    img_files = sorted(glob.glob(os.path.join(image_dir, f"img[1-5].jpg")))
    
    if len(img_files) != 5:
        # Try other naming patterns
        img_files = sorted(glob.glob(os.path.join(image_dir, f"*.jpg")))
        img_files = img_files[:5]
        if len(img_files) != 5:
            raise ValueError(f"Need 5 images, found {len(img_files)}! Please ensure 5 jpg images in folder")
    
    print(f"Found {len(img_files)} image files")

    # Load first image to determine size
    first_img = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)
    h_ref, w_ref = first_img.shape
    print(f"Original image size: {h_ref}×{w_ref}")

    # Preprocessing and cropping
    h1, h2 = int(h_ref*0.1), int(h_ref*0.9)
    w1, w2 = int(w_ref*0.1), int(w_ref*0.9)

    imgs = []
    for i, f in enumerate(img_files, 1):
        print(f"Processing image {i}: {os.path.basename(f)}")
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        # Crop image
        img_crop = img[h1:h2, w1:w2]
        
        # Scale image
        h_crop, w_crop = img_crop.shape
        h_scaled, w_scaled = int(h_crop * scale_factor), int(w_crop * scale_factor)
        img_resized = cv2.resize(img_crop, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
        
        # Normalize
        img_norm = img_resized / 255.0

        # Background filtering (keep only object area)
        mean_brightness = np.mean(img_norm)
        mask = img_norm > mean_brightness * 1.5
        img_norm[~mask] = 0.0
        imgs.append(img_norm)
        print(f"  Cropped size: {h_crop}×{w_crop}, Scaled: {h_scaled}×{w_scaled}")

    h, w = imgs[0].shape
    print(f"Final processing size: {h}×{w} (approx {h*w:,} pixels)")

    # Calculate pseudoinverse
    L_pinv = np.linalg.pinv(L)
    print("Light matrix pseudoinverse calculated")

    # Initialize results
    normal_map = np.zeros((h, w, 3), dtype=np.float32)
    albedo_map = np.zeros((h, w), dtype=np.float32)
    mask_valid = np.zeros((h, w), dtype=bool)  # Record valid pixel area

    # Multi-thread calculation
    print("Calculating normals and albedo...")
    imgs_2d = np.stack(imgs, axis=0)
    
    # Add progress indicator
    total_pixels = h * w
    processed = 0
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_pixel, y, x, imgs_2d, L_pinv)
                   for y in range(h) for x in range(w)]
        
        for i, future in enumerate(futures):
            y, x, normal, rho = future.result()
            normal_map[y, x] = normal
            albedo_map[y, x] = rho
            if rho > 1e-4:
                mask_valid[y, x] = True
            
            # Progress indicator
            processed += 1
            if processed % 100000 == 0 or processed == total_pixels:
                print(f"  Progress: {processed}/{total_pixels} ({processed/total_pixels*100:.1f}%)")

    print(f"Multi-thread calculation complete. Valid pixels: {mask_valid.sum()}/{total_pixels} ({mask_valid.sum()/total_pixels*100:.1f}%)")

    # Save visualization results - block processing to avoid memory overflow
    print("Generating visualization results...")
    
    # Block processing for normal map
    h, w, c = normal_map.shape
    normal_vis = np.zeros((h, w, c), dtype=np.uint8)
    block_size = 256
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            i_end = min(i + block_size, h)
            j_end = min(j + block_size, w)
            
            # Extract block
            block = normal_map[i:i_end, j:j_end]
            
            # Convert and scale
            block_vis = ((block + 1.0) * 127.5)
            
            # Ensure values are in valid range
            np.clip(block_vis, 0, 255, out=block_vis)
            
            # Assign and convert type
            normal_vis[i:i_end, j:j_end] = block_vis.astype(np.uint8)
            
            # Clean up
            del block, block_vis
    
    # Process albedo map
    albedo_min = np.min(albedo_map[mask_valid])
    albedo_max = np.max(albedo_map[mask_valid])
    albedo_normalized = np.zeros_like(albedo_map)
    albedo_normalized[mask_valid] = (albedo_map[mask_valid] - albedo_min) / (albedo_max - albedo_min + 1e-8)
    
    # Convert to 8-bit and apply Gaussian blur
    albedo_vis = (albedo_normalized * 255).astype(np.uint8)
    albedo_vis = cv2.GaussianBlur(albedo_vis, (3, 3), 0)
    
    cv2.imwrite(os.path.join(save_dir, f"normal_map_{scene_name}.png"), normal_vis)
    cv2.imwrite(os.path.join(save_dir, f"albedo_map_{scene_name}.png"), albedo_vis)
    print(f"Normal map and albedo map saved")

    # Save raw data
    np.save(os.path.join(save_dir, f"normal_map_float_{scene_name}.npy"), normal_map)
    np.save(os.path.join(save_dir, f"albedo_map_float_{scene_name}.npy"), albedo_map)
    np.save(os.path.join(save_dir, f"mask_valid_{scene_name}.npy"), mask_valid)
    # Save light matrix
    np.savetxt(os.path.join(save_dir, f"light_matrix_{scene_name}.txt"), L, fmt="%.4f")
    print(f"Raw float data saved")

    print(f"✓ Photometric stereo processing complete! Results saved to: {save_dir}")
    return normal_map, albedo_map, mask_valid, (h, w)


# ==================== Part 2: Depth Map Reconstruction ====================

def compute_gradients(normal_map, mask_valid):
    """
    Calculate depth gradients from surface normals
    """
    print("Calculating depth gradients...")
    n_x = normal_map[..., 0]
    n_y = normal_map[..., 1]
    n_z = normal_map[..., 2]
    
    # Safe division
    nz_safe = np.where(np.abs(n_z) < 1e-8, 1e-8, n_z)
    
    # Calculate gradients p = ∂Z/∂x, q = ∂Z/∂y
    p = -n_x / nz_safe
    q = -n_y / nz_safe
    
    # Keep only valid areas, set background to zero
    p[~mask_valid] = 0
    q[~mask_valid] = 0
    
    print("Depth gradient calculation complete")
    return p, q

def frankot_chellappa_integrate(p, q, mask_valid):
    """
    Frankot-Chellappa global optimal integration method
    """
    print("Using Frankot-Chellappa method for depth integration...")
    h, w = p.shape
    
    # Create frequency coordinates
    u = fftfreq(w)
    v = fftfreq(h)
    U, V = np.meshgrid(u, v)
    
    # Frequency domain divisor
    denom = (2j * np.pi) * (U + 1j * V)
    denom[0, 0] = 1.0
    
    # FFT transform
    P = fft2(p)
    Q = fft2(q)
    
    # Frequency domain integration
    Z_freq = (U * P + V * Q) / (denom + 1e-8)
    Z_freq[0, 0] = 0  # Remove DC component
    
    # Inverse transform
    depth = ifft2(Z_freq).real
    
    # Keep depth values only in valid areas
    depth[~mask_valid] = np.nan
    
    print("Frankot-Chellappa integration complete")
    return depth

def poisson_integrate(p, q, mask_valid, iterations=1000):
    """
    Poisson reconstruction (iterative method)
    """
    print(f"Using Poisson method for depth integration (iterations={iterations})...")
    h, w = p.shape
    depth = np.zeros((h, w), dtype=np.float32)
    
    # Calculate divergence
    div = np.zeros((h, w), dtype=np.float32)
    div[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2] + 
                       q[2:, 1:-1] - q[:-2, 1:-1]) / 2
    
    # Jacobi iteration
    for it in range(iterations):
        depth_new = np.zeros_like(depth)
        depth_new[1:-1, 1:-1] = (
            depth[:-2, 1:-1] + depth[2:, 1:-1] + 
            depth[1:-1, :-2] + depth[1:-1, 2:] - div[1:-1, 1:-1]
        ) / 4
        
        # Maintain boundary conditions
        depth_new[~mask_valid] = 0
        depth = depth_new
        
        # Progress indicator
        if (it + 1) % 100 == 0 or it == 0 or it == iterations - 1:
            print(f"  Poisson iteration progress: {it+1}/{iterations}")
    
    print("Poisson integration complete")
    return depth

def normalize_depth_for_display(depth, mask_valid, method='robust'):
    """
    Normalize depth map for display
    """
    print("Normalizing depth map...")
    # Extract valid depth values
    valid_depth = depth[mask_valid]
    if len(valid_depth) == 0:
        print("Warning: No valid depth values!")
        return np.zeros_like(depth)
    
    if method == 'robust':
        # Robust normalization (remove outliers)
        d_min = np.percentile(valid_depth, 1)
        d_max = np.percentile(valid_depth, 99)
    else:  # Simple normalization
        d_min = np.nanmin(valid_depth)
        d_max = np.nanmax(valid_depth)
    
    print(f"Depth value range: {d_min:.4f} to {d_max:.4f}")
    
    # Normalize to 0-255
    depth_normalized = np.zeros_like(depth)
    depth_normalized[mask_valid] = (valid_depth - d_min) / (d_max - d_min + 1e-8) * 255
    
    # Set invalid areas to 0
    depth_normalized[~mask_valid] = 0
    
    print("Depth map normalization complete")
    return depth_normalized.astype(np.uint8)

def reconstruct_depth(normal_map, mask_valid, save_dir, scene_name, method='frankot_chellappa'):
    """
    Depth map reconstruction main function
    
    Parameters:
    normal_map: Normal map
    mask_valid: Valid area mask
    save_dir: Save directory
    scene_name: Scene name
    method: Integration method, 'frankot_chellappa' or 'poisson'
    """
    print(f"================ Starting Depth Map Reconstruction ================")
    print(f"Scene: {scene_name}")
    print(f"Integration method: {method}")
    
    # Calculate depth gradients
    p, q = compute_gradients(normal_map, mask_valid)
    
    # Select integration method
    if method == 'frankot_chellappa':
        depth_raw = frankot_chellappa_integrate(p, q, mask_valid)
    elif method == 'poisson':
        depth_raw = poisson_integrate(p, q, mask_valid, iterations=500)
    else:
        raise ValueError(f"Unknown integration method: {method}")
    
    # Normalize depth map for display
    depth_display = normalize_depth_for_display(depth_raw, mask_valid, method='robust')
    
    # Save results
    depth_path = os.path.join(save_dir, f"depth_map_{scene_name}_{method}.png")
    cv2.imwrite(depth_path, depth_display)
    print(f"Depth map saved: {depth_path}")
    
    # Save raw depth data
    depth_raw_path = os.path.join(save_dir, f"depth_raw_{scene_name}_{method}.npy")
    np.save(depth_raw_path, depth_raw)
    print(f"Raw depth data saved: {depth_raw_path}")
    
    print(f"✓ Depth map reconstruction complete!")
    return depth_raw, depth_display


# ==================== Main Program ====================

if __name__ == "__main__":
    # Configuration parameters
    SCENE_NAME = "s3"  # Scene name: "s1", "s2" or "s3"
    SCALE_FACTOR = 0.25  # Image scaling factor
    
    # Generate paths based on scene name
    IMAGE_DIR = f"images_{SCENE_NAME}"
    SAVE_DIR = f"results_{SCENE_NAME}"
    
    try:
        print("=" * 50)
        print("Photometric Stereo + Depth Reconstruction System (5 lights)")
        print("=" * 50)
        print(f"Processing scene: {SCENE_NAME}")
        print(f"Image directory: {IMAGE_DIR}")
        print(f"Save directory: {SAVE_DIR}")
        print(f"Scale factor: {SCALE_FACTOR}")
        print("=" * 50)
        
        # Step 1: Photometric stereo processing
        normal_map, albedo_map, mask_valid, img_shape = photometric_stereo_scene(
            IMAGE_DIR, SAVE_DIR, SCALE_FACTOR, SCENE_NAME
        )
        
        # Step 2: Depth map reconstruction
        depth_raw, depth_display = reconstruct_depth(
            normal_map, mask_valid, SAVE_DIR, SCENE_NAME, method='frankot_chellappa'
        )
        
        print("=" * 50)
        print("✅ Processing complete!")
        print(f"Processed scene: {SCENE_NAME}")
        print(f"Results saved to: {SAVE_DIR}")
        print("\nGenerated files:")
        print(f"  - normal_map_{SCENE_NAME}.png")
        print(f"  - albedo_map_{SCENE_NAME}.png")
        print(f"  - depth_map_{SCENE_NAME}_frankot_chellappa.png")
        print(f"  - normal_map_float_{SCENE_NAME}.npy")
        print(f"  - albedo_map_float_{SCENE_NAME}.npy")
        print(f"  - mask_valid_{SCENE_NAME}.npy")
        print(f"  - depth_raw_{SCENE_NAME}_frankot_chellappa.npy")
        print(f"  - light_matrix_{SCENE_NAME}.txt")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Program error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 50)
        print("Program terminated abnormally")