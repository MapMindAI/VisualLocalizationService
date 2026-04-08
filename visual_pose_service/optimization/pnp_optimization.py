# Copyright 2025 DeepMirror Inc. All rights reserved.

import numpy as np
import cv2
from scipy.optimize import least_squares


def project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs=None):
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)


def reprojection_error_with_gravity(
    params,
    points_3d,
    points_2d,
    camera_matrix,
    dist_coeffs,
    gravity_world,
    gravity_camera_expected,
    gravity_weight=1.0,
):
    rvec = params[:3]
    tvec = params[3:6]

    # compute reprojection error
    projected_points = project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    reprojection_residual = (projected_points - points_2d).ravel()

    # compute rotation matrix -> world to camera
    R_mat, _ = cv2.Rodrigues(rvec)

    # compute gravity residual
    # R * g_world should be close to g_camera_expected
    gravity_camera_actual = R_mat @ gravity_world
    # gravity_residual = gravity_weight * (gravity_camera_actual - gravity_camera_expected)
    gravity_residual = gravity_weight * (1 - np.dot(gravity_camera_actual, gravity_camera_expected))
    gravity_residual = np.array([gravity_residual])

    # get all the residual
    return np.concatenate([reprojection_residual, gravity_residual])


def pnp_with_gravity(
    points_3d,
    points_2d,
    camera_matrix,
    dist_coeffs,
    gravity_world,
    gravity_camera_expected,
    gravity_weight=1.0,
    initial_rvec=None,
    initial_tvec=None,
):
    if initial_rvec is None or initial_tvec is None:
        success, rvec, tvec = cv2.solvePnP(
            points_3d, points_2d, camera_matrix, None, flags=cv2.SOLVEPNP_EPNP
        )
        if not success:
            raise RuntimeError("Initial PnP estimation failed.")
    else:
        rvec = initial_rvec
        tvec = initial_tvec

    initial_params = np.hstack((rvec.ravel(), tvec.ravel()))

    # For many points, use method='trf' or 'dogbox' to avoid LM equation limits.
    result = least_squares(
        reprojection_error_with_gravity,
        initial_params,
        method='dogbox',
        loss='huber',  # 'soft_l1', 'huber', 'cauchy', 'arctan'
        args=(
            points_3d,
            points_2d,
            camera_matrix,
            dist_coeffs,
            gravity_world,
            gravity_camera_expected,
            gravity_weight,
        ),
        verbose=2,
    )

    optimized_rvec = result.x[:3].reshape(3, 1)
    optimized_tvec = result.x[3:6].reshape(3, 1)

    return optimized_rvec, optimized_tvec


def pnp_ransac_with_gravity(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs=None,
    gravity_world=None,
    gravity_camera_expected=None,
    gravity_angle_error_threshold_degree=20,
    max_iterations=100,
    reprojection_error_threshold=8.0,
    min_inliers=6,
    confidence=0.99,
    sample_size=6,
):
    """
    Custom PnP RANSAC using solvePnP with SOLVEPNP_EPNP and adaptive early exit.

    Parameters:
        object_points (np.ndarray): 3D points, shape (N, 3)
        image_points (np.ndarray): 2D image points, shape (N, 2)
        camera_matrix (np.ndarray): Camera intrinsics (3x3)
        dist_coeffs (np.ndarray or None): Distortion coefficients
        max_iterations (int): Initial max number of RANSAC iterations
        reprojection_error_threshold (float): Inlier threshold in pixels
        min_inliers (int): Minimum number of inliers to accept a model
        confidence (float): Desired success confidence (0–1)
        sample_size (int): Number of points per minimal sample (≥4 for EPNP)

    Returns:
        best_rvec, best_tvec, best_inliers (indices)
    """
    assert object_points.shape[0] == image_points.shape[0]
    num_points = object_points.shape[0]
    if num_points < sample_size:
        return False, None, None, [], 0
    # assert num_points >= sample_size, "Not enough points for sampling"

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))

    best_num_inliers = 0
    best_rvec, best_tvec = None, None
    best_inliers = []

    iterations = 0
    log_one_minus_conf = np.log(1 - confidence)

    while iterations < max_iterations:
        # Sample
        idx = np.random.choice(num_points, sample_size, replace=False)
        obj_sample = object_points[idx]
        img_sample = image_points[idx]

        # Estimate pose using EPNP
        success, rvec, tvec = cv2.solvePnP(
            obj_sample, img_sample, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
        )
        if not success:
            iterations += 1
            continue

        # Reproject all 3D points
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.squeeze()
        errors = np.linalg.norm(projected - image_points, axis=1)
        inliers = np.where(errors < reprojection_error_threshold)[0]
        cur_inlier_size = len(inliers)

        # check gravity error
        if (gravity_world is not None) and (gravity_camera_expected is not None):
            R_mat, _ = cv2.Rodrigues(rvec)
            gravity_camera_actual = R_mat @ gravity_world
            gravity_error = np.degrees(
                np.arccos(np.dot(gravity_camera_actual, gravity_camera_expected))
            )
            if abs(gravity_error) > gravity_angle_error_threshold_degree:
                print(f"iteration {iterations},  gravity error too large: {gravity_error:.2f} deg")
                iterations += 1
                continue

        # Update best if more inliers
        if cur_inlier_size > best_num_inliers:
            best_num_inliers = cur_inlier_size
            best_rvec, best_tvec = rvec, tvec
            best_inliers = inliers

            # Early exit: update max_iterations if possible
            inlier_ratio = best_num_inliers / num_points
            success_iter_prob = inlier_ratio**sample_size

            if success_iter_prob < -log_one_minus_conf / max_iterations:
                iterations += 1
                continue  # skip numeric instability

            needed_iterations = log_one_minus_conf / np.log(1 - success_iter_prob)
            if needed_iterations < max_iterations:
                max_iterations = int(np.ceil(needed_iterations))

        iterations += 1

    if best_rvec is None:
        # raise RuntimeError("PnP RANSAC failed to find a valid pose")
        return False, None, None, [], iterations

    return True, best_rvec, best_tvec, best_inliers, iterations


def pnp_ransac_optimization_with_gravity(
    matched_3d,
    matched_2d,
    camera_matrix,
    dist_coeffs,
    gravity_world,
    gravity_camera_expected,
    gravity_weight,
    gravity_angle_error_threshold_degree,
    ransac_reproj_error,
    ransac_iterations=500,
    ransac_confidence=0.95,
):
    if gravity_camera_expected is None or np.linalg.norm(gravity_camera_expected) == 0.0:
        # process pnp
        success, rot_vec, tran_vec, inliers = cv2.solvePnPRansac(
            matched_3d,
            matched_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=ransac_reproj_error,
            iterationsCount=ransac_iterations,
            confidence=ransac_confidence,
        )
        return success, rot_vec, tran_vec, inliers

    grav_norm = np.linalg.norm(gravity_camera_expected)
    if grav_norm == 0.0:
        return False, None, None, []

    # normalize gravity
    gravity_world_n = gravity_world / np.linalg.norm(gravity_world)
    gravity_camera_expected_n = gravity_camera_expected / grav_norm

    success, rot_vec, tran_vec, inliers, iterations = pnp_ransac_with_gravity(
        matched_3d,
        matched_2d,
        camera_matrix,
        dist_coeffs=dist_coeffs,
        gravity_world=gravity_world_n,
        gravity_camera_expected=gravity_camera_expected_n,
        gravity_angle_error_threshold_degree=gravity_angle_error_threshold_degree,
        max_iterations=ransac_iterations,
        reprojection_error_threshold=ransac_reproj_error,
        min_inliers=6,
        confidence=ransac_confidence,
        sample_size=6,
    )

    # collect the inliers
    # Convert inliers index list into flat list
    if not success:
        return False, None, None, []

    inlier_idxs = inliers.flatten()

    # Extract inlier 2D-3D points
    inlier_points_3d = matched_3d[inlier_idxs]
    inlier_points_2d = matched_2d[inlier_idxs]

    # process optimization with gravity
    rvec_opt, tvec_opt = pnp_with_gravity(
        inlier_points_3d,  # ← Only inliers
        inlier_points_2d,
        camera_matrix,
        dist_coeffs,
        gravity_world=gravity_world_n,  # Defined in world frame
        gravity_camera_expected=gravity_camera_expected_n,  # Expected in camera frame
        gravity_weight=gravity_weight,
        initial_rvec=rot_vec,
        initial_tvec=tran_vec,
    )

    return True, rvec_opt, tvec_opt, inliers


def test_pnp_with_gravity():
    print("test_pnp_with_gravity:")
    num_points = 20
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
    points_3d = []
    for i in range(num_points):
        points_3d.append(5.0 * np.random.rand(3, 1))
    points_3d = np.array(points_3d, dtype=np.float64)
    # Ground-truth pose
    true_rvec = np.array([[0.1], [0.2], [0.3]])
    true_tvec = np.array([[0.5], [0.2], [1.0]])
    true_R_mat, _ = cv2.Rodrigues(true_rvec)

    # Project to image
    points_2d = project_points(points_3d, true_rvec, true_tvec, camera_matrix)

    # Add noise
    noise = np.random.normal(0, 5, size=points_2d.shape)
    points_2d_noisy = points_2d + noise

    gravity_world = np.array([0, 0, -1])
    gravity_camera_expected = true_R_mat @ gravity_world

    # Gravity term weight (larger = stronger)
    gravity_weight = 10.0

    rvec_opt, tvec_opt = pnp_with_gravity(
        points_3d,
        points_2d_noisy,
        camera_matrix,
        None,
        gravity_world,
        gravity_camera_expected,
        gravity_weight=gravity_weight,
        initial_rvec=np.random.rand(3, 1),
        initial_tvec=np.random.rand(3, 1),
    )

    print(f"  after optimization {rvec_opt.transpose()} {tvec_opt.transpose()}")
    print(f"  pose error {(rvec_opt - true_rvec).transpose()} {(tvec_opt - true_tvec).transpose()}")


def test_pnp_ransac_with_gravity():
    print("test_pnp_ransac_with_gravity")
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
    num_points = 50
    num_outliers = int(num_points * 0.2)

    points_3d = []
    for i in range(num_points):
        points_3d.append(5.0 * np.random.rand(3, 1))
    points_3d = np.array(points_3d, dtype=np.float64)

    true_rvec = np.array([[0.1], [0.2], [0.3]])
    true_tvec = np.array([[0.5], [0.2], [1.0]])
    true_R_mat, _ = cv2.Rodrigues(true_rvec)
    points_2d = project_points(points_3d, true_rvec, true_tvec, camera_matrix)

    # add outlier
    for i in range(num_outliers):
        # randomly assign point 2d to be outlier
        points_2d[i] = 600.0 * np.random.rand(2)

    # add noise
    noise = np.random.normal(0, 0.5, size=points_2d.shape)
    points_2d_noisy = points_2d + noise

    gravity_world = np.array([0, 0, -1])
    gravity_camera_expected = true_R_mat @ gravity_world

    # Gravity term weight (larger = stronger)
    gravity_weight = 10.0

    success, rvec_opt, tvec_opt, best_inliers, iterations = pnp_ransac_with_gravity(
        points_3d,
        points_2d_noisy,
        camera_matrix,
        dist_coeffs=None,
        gravity_world=gravity_world,
        gravity_camera_expected=gravity_camera_expected,
        gravity_angle_error_threshold_degree=20,
        max_iterations=100,
        reprojection_error_threshold=8.0,
        min_inliers=6,
        confidence=0.99,
        sample_size=6,
    )

    print(
        f"  ransac {rvec_opt.transpose()} {tvec_opt.transpose()}, #inliers: {len(best_inliers)} #iterations: {iterations}"
    )
    print(f"  error {(rvec_opt - true_rvec).transpose()} {(tvec_opt - true_tvec).transpose()}")


if __name__ == "__main__":
    np.random.seed(42)
    test_pnp_with_gravity()
    test_pnp_ransac_with_gravity()
