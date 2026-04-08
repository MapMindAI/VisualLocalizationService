"""
Homography-based image rotation (concept summary)
=================================================
1. Why a homography?
   - We estimate a 3D rotation in the camera frame.
   - The image is 2D; we map that 3D rotation to the image plane.
   - H = K * R * K^(-1) implements this mapping.

2. How H acts
   - K: intrinsics, project 3D to 2D.
   - R: 3D rotation of the camera frame.
   - K^(-1): back-project pixels to the normalized plane.
   - H = K * R * K^(-1): equivalent 2D warp.
   - Meaning: rotate the camera **in place** (no translation).
   - Assumption: planar scene or depth variation << distance (valid for flat markers).

   Per pixel [u, v]:
     a) Back-project: [X_norm, Y_norm, 1] = K^(-1) * [u, v, 1]
     b) Rotate: [X_norm', Y_norm', 1] = R * [X_norm, Y_norm, 1]
     c) Project: [u', v', 1] = K * [X_norm', Y_norm', 1]
     Same as [u', v', 1] = (K * R * K^(-1)) * [u, v, 1]

3. Pixel warp
   For homogeneous p = [u, v, 1]^T: p' = H * p, then interpolate.

4. cv2.warpPerspective
   For each output pixel (x', y'), sample H^(-1) * [x', y', 1]^T in the source image.
"""

import numpy as np
import cv2
import sys
import os
from scipy.spatial.transform import Rotation as R

def unreal_to_slam(rotation_matrix, translation=None):
    """
    Convert pose from Unreal coordinate system to SLAM coordinate system.
    
    Args:
        rotation_matrix: 3x3 rotation matrix
        translation: 3x1 translation vector (optional, defaults to zero)
    
    Returns:
        tuple: (transformed_rotation_matrix, transformed_translation)
    """
    # Transformation matrix to switch between Unreal and SLAM coordinate systems
    # Matrix: [[0, 1, 0],
    #          [0, 0, -1],
    #          [1, 0, 0]]
    transformation_matrix = np.array([
        [0, 1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ], dtype=np.float64)
    
    # Transform rotation: R_slam = T * R_unreal * T^T
    transformed_rotation = transformation_matrix @ rotation_matrix @ transformation_matrix.T
    
    # Transform translation: t_slam = T * t_unreal
    if translation is not None:
        transformed_translation = transformation_matrix @ translation
    else:
        transformed_translation = np.zeros(3, dtype=np.float64)
    
    return transformed_rotation, transformed_translation

# Input parameters
image_path = '/Mobili/Data/record/20251125150423/vlp_output/images/1033533976017463.jpg'
image = cv2.imread(image_path)

# Camera pose (camera_to_local): quaternion [w, x, y, z] and translation [x, y, z]
# Note: This pose is in Unreal coordinate system and needs to be converted to SLAM
w, x, y, z = 0.946308970, 0.001175920, 0.323199004, 0.006349718
quat_xyzw = np.array([x, y, z, w])
rot_unreal = R.from_quat(quat_xyzw)
R_unreal = rot_unreal.as_matrix()

print("Original rotation matrix (Unreal):")
print(R_unreal)

# Convert from Unreal to SLAM coordinate system
R_slam, t_slam = unreal_to_slam(R_unreal)
rot = R.from_matrix(R_slam)
print("\nConverted rotation matrix (SLAM):")
print(R_slam)

# Camera intrinsics
fx, fy, cx, cy = 325.610229492, 325.626342773, 255.875000000, 191.875000000
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Marker normal in local frame (parallel to gravity direction)
# Gravity direction in local frame: (0, 1, 0) (from pose visualization)
# Marker plane is perpendicular to gravity, so marker normal = gravity direction
gravity_local = np.array([0, 1, 0])

print("=== Image Transformation: Marker Plane to Camera Plane ===")

# Step 1: Transform marker normal from local to camera frame
print(f"\nStep 1: Transform marker normal from local to camera frame")
R_cam_to_local = R_slam  # Now in SLAM coordinate system
R_local_to_cam = R_cam_to_local.T
marker_normal_camera = R_local_to_cam @ gravity_local
marker_normal_camera = marker_normal_camera / np.linalg.norm(marker_normal_camera)
print(f"  Marker normal (camera): {marker_normal_camera}")

# Step 2: Compute rotation to align marker normal to camera z-axis
print(f"\nStep 2: Compute rotation to align marker normal to camera z-axis")
# Target: marker normal should be parallel to camera z-axis (0, 0, 1)
marker_normal_target = np.array([0, 0, 1])
rotation_axis = np.cross(marker_normal_camera, marker_normal_target)
axis_norm = np.linalg.norm(rotation_axis)

if axis_norm < 1e-6:
    if np.dot(marker_normal_camera, marker_normal_target) > 0.99:
        R_rotation = np.eye(3)
        angle = 0.0
    else:
        if abs(marker_normal_camera[0]) < 0.9:
            rotation_axis = np.cross(marker_normal_camera, np.array([1, 0, 0]))
        else:
            rotation_axis = np.cross(marker_normal_camera, np.array([0, 1, 0]))
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.pi
else:
    rotation_axis = rotation_axis / axis_norm
    cos_angle = np.clip(np.dot(marker_normal_camera, marker_normal_target), -1.0, 1.0)
    angle = np.arccos(cos_angle)

print(f"  Rotation angle: {np.degrees(angle):.2f} degrees")

# Create rotation matrix using Rodrigues' formula
if angle < 1e-6:
    R_rotation = np.eye(3)
else:
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    R_rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

# Step 3: Compute homography H = K * R * K^(-1) and warp image
print("\nStep 3: Compute homography H = K * R * K^(-1) and warp image")
h, w = image.shape[:2]

# R_rotation rotates marker_normal_camera to [0, 0, 1] (rotates the vector)
# To see the rotated view, we need to apply inverse rotation to the camera frame
# So we use R_rotation.T (inverse rotation) in the homography
# H = K * R^(-1) * K^(-1) where R^(-1) = R_rotation.T
K_inv = np.linalg.inv(camera_matrix)
R_inv = R_rotation.T  # Inverse rotation
H = camera_matrix @ R_rotation @ K_inv
H = H / H[2, 2]  # Normalize

print(f"Homography matrix:\n{H}")

# Transform image corners to determine output size
print("\nComputing output image size:")
print("1. Warp the four image corners with H")
corners_original = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
print(f"   Original corners: {corners_original.reshape(-1, 2)}")
corners_transformed = cv2.perspectiveTransform(corners_original, H).reshape(-1, 2)
print(f"   Warped corners: {corners_transformed}")
x_min, x_max = int(np.floor(corners_transformed[:, 0].min())), int(np.ceil(corners_transformed[:, 0].max()))
y_min, y_max = int(np.floor(corners_transformed[:, 1].min())), int(np.ceil(corners_transformed[:, 1].max()))

# Add margin and adjust homography for translation
margin = 50
x_min, y_min = min(x_min, -margin), min(y_min, -margin)
x_max, y_max = max(x_max, w + margin), max(y_max, h + margin)

print(f"2. Bounding box: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
print("3. Add translation so all pixels stay in positive coordinates")

translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
H_final = translation @ H
output_width, output_height = x_max - x_min, y_max - y_min

print(f"Output image size: {output_width}x{output_height}")

# Apply transformation
image_transformed = cv2.warpPerspective(image, H_final, (output_width, output_height), 
                                        flags=cv2.INTER_LINEAR, 
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))

# Save result
output_dir = os.path.dirname(image_path)
base_name = os.path.splitext(os.path.basename(image_path))[0]
output_path = os.path.join(output_dir, f"{base_name}_transformed_by_gravity.jpg")
cv2.imwrite(output_path, image_transformed)
print(f"\nTransformed image saved to: {output_path}")

# ========== Step 4: SIFT Feature Matching ==========
print("\n" + "=" * 60)
print("Step 4: SIFT Feature Matching")
print("=" * 60)

# Load marker image (you need to specify the path)
marker_image_path = '/Mobili/python/visual_pose_service/marker_pose_service/marker_image/dm_final.jpg'  # TODO: Update this path
if not os.path.exists(marker_image_path):
    print(f"Warning: Marker image not found at {marker_image_path}")
    print("Please update marker_image_path in the script")
    marker_image = None
else:
    marker_image = cv2.imread(marker_image_path)
    if marker_image is None:
        print(f"Warning: Failed to load marker image from {marker_image_path}")
        marker_image = None

good_matches = []  # Initialize for later use
img_matches_transformed = None
img_original_with_points = None

if marker_image is not None:
    # Initialize SIFT
    sift = cv2.SIFT_create()
    
    # Extract features from transformed image
    print("\n4.1: Extracting SIFT features from transformed image...")
    gray_transformed = cv2.cvtColor(image_transformed, cv2.COLOR_BGR2GRAY)
    kp_transformed, desc_transformed = sift.detectAndCompute(gray_transformed, None)
    print(f"  Found {len(kp_transformed)} keypoints in transformed image")
    
    # Extract features from marker image
    print("\n4.2: Extracting SIFT features from marker image...")
    gray_marker = cv2.cvtColor(marker_image, cv2.COLOR_BGR2GRAY)
    kp_marker, desc_marker = sift.detectAndCompute(gray_marker, None)
    print(f"  Found {len(kp_marker)} keypoints in marker image")
    
    if desc_transformed is not None and desc_marker is not None and len(desc_transformed) > 0 and len(desc_marker) > 0:
        # Feature matching using FLANN
        print("\n4.3: Matching features...")
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(desc_transformed, desc_marker, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        print(f"  Found {len(good_matches)} good matches (after ratio test)")
        
        if len(good_matches) > 0:
            # Get matched keypoints in transformed image
            matched_kp_transformed = [kp_transformed[m.queryIdx] for m in good_matches]
            matched_kp_marker = [kp_marker[m.trainIdx] for m in good_matches]
            
            # Extract coordinates from transformed image
            matched_points_transformed = np.array([[kp.pt[0], kp.pt[1]] for kp in matched_kp_transformed], dtype=np.float32)
            
            print(f"\n4.4: Transforming matched points back to original image coordinates...")
            print(f"  Matched points in transformed image: {len(matched_points_transformed)}")
            
            # Transform points back to original image coordinates
            # Note: H_final includes translation, so we need to use its inverse
            H_final_inv = np.linalg.inv(H_final)
            
            # Convert to homogeneous coordinates
            matched_points_transformed_homogeneous = np.hstack([
                matched_points_transformed, 
                np.ones((len(matched_points_transformed), 1), dtype=np.float32)
            ])
            
            # Transform back to original image coordinates
            matched_points_original_homogeneous = (H_final_inv @ matched_points_transformed_homogeneous.T).T
            matched_points_original = matched_points_original_homogeneous[:, :2] / matched_points_original_homogeneous[:, 2:3]
            
            print(f"  Matched points in original image: {len(matched_points_original)}")
            
            # Filter points that are within original image bounds
            valid_mask = (
                (matched_points_original[:, 0] >= 0) & 
                (matched_points_original[:, 0] < w) &
                (matched_points_original[:, 1] >= 0) & 
                (matched_points_original[:, 1] < h)
            )
            matched_points_original_valid = matched_points_original[valid_mask]
            matched_kp_marker_valid = [matched_kp_marker[i] for i in range(len(matched_kp_marker)) if valid_mask[i]]
            
            print(f"  Valid points (within original image bounds): {len(matched_points_original_valid)}")
            
            # Visualize matches
            print("\n4.5: Visualizing matches...")
            
            # Draw matches on transformed image
            img_matches_transformed = cv2.drawMatches(
                image_transformed, kp_transformed,
                marker_image, kp_marker,
                good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            # Draw matches on original image
            # Create a visualization showing original image with transformed points
            img_original_with_points = image.copy()
            for pt in matched_points_original_valid:
                cv2.circle(img_original_with_points, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                cv2.circle(img_original_with_points, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), 2)
            
            # Also draw corresponding marker points
            img_marker_with_points = marker_image.copy()
            for kp in matched_kp_marker_valid:
                cv2.circle(img_marker_with_points, (int(kp.pt[0]), int(kp.pt[1])), 5, (0, 255, 0), -1)
                cv2.circle(img_marker_with_points, (int(kp.pt[0]), int(kp.pt[1])), 10, (0, 255, 0), 2)
            
            # Save visualization
            matches_vis_path = os.path.join(output_dir, f"{base_name}_matches_transformed.jpg")
            cv2.imwrite(matches_vis_path, img_matches_transformed)
            print(f"  Saved matches visualization (transformed image) to: {matches_vis_path}")
            
            original_points_vis_path = os.path.join(output_dir, f"{base_name}_original_with_points.jpg")
            cv2.imwrite(original_points_vis_path, img_original_with_points)
            print(f"  Saved original image with transformed points to: {original_points_vis_path}")
            
            marker_points_vis_path = os.path.join(output_dir, f"{base_name}_marker_with_points.jpg")
            cv2.imwrite(marker_points_vis_path, img_marker_with_points)
            print(f"  Saved marker image with matched points to: {marker_points_vis_path}")
            
            # Save matched points to file
            points_file_path = os.path.join(output_dir, f"{base_name}_matched_points.txt")
            with open(points_file_path, 'w') as f:
                f.write("# Matched feature points\n")
                f.write("# Format: original_x, original_y, transformed_x, transformed_y, marker_x, marker_y\n")
                for i, (pt_orig, pt_trans, kp_m) in enumerate(zip(
                    matched_points_original_valid, 
                    matched_points_transformed[valid_mask],
                    matched_kp_marker_valid
                )):
                    f.write(f"{pt_orig[0]:.2f}, {pt_orig[1]:.2f}, "
                           f"{pt_trans[0]:.2f}, {pt_trans[1]:.2f}, "
                           f"{kp_m.pt[0]:.2f}, {kp_m.pt[1]:.2f}\n")
            print(f"  Saved matched points to: {points_file_path}")
            
            print("\n" + "=" * 60)
            print("Summary:")
            print("=" * 60)
            print(f"  Original image size: {w}x{h}")
            print(f"  Transformed image size: {output_width}x{output_height}")
            print(f"  Matched feature points: {len(matched_points_original_valid)}")
            print(f"  Matched points saved to: {points_file_path}")
            print("=" * 60)
        else:
            print("  No good matches found after ratio test")
    else:
        print("  Failed to extract features from one or both images")
else:
    print("  Skipping feature matching (marker image not available)")

# Display results
print("\nDisplaying results...")
cv2.imshow('Original Image', image)
cv2.imshow('Transformed Image', image_transformed)
if marker_image is not None and len(good_matches) > 0 and img_matches_transformed is not None:
    cv2.imshow('Matches (Transformed Image)', img_matches_transformed)
if img_original_with_points is not None:
    cv2.imshow('Original Image with Points', img_original_with_points)
print("\nPress any key to close windows...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n=== Done ===")
