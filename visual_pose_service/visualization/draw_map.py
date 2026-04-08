import numpy as np
import open3d as o3d
import argparse
import os
import cv2
import sqlite3
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

MESH = None


def project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs):
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    points_2d = points_2d.reshape(-1, 2)
    return points_2d


def draw_points(image, points_2d, color=(0, 255, 0), radius=4):
    height, width = image.shape[:2]
    for point in points_2d:
        pt = tuple(map(int, point))
        if 0 <= pt[0] < width and 0 <= pt[1] < height:
            cv2.circle(image, pt, radius, color, -1)


def draw_lines(image, points1, points2, color=(0, 0, 255), thickness=3):
    for i in range(len(points1)):
        pt1 = tuple(map(int, points1[i]))
        pt2 = tuple(map(int, points2[i]))
        cv2.line(image, pt1, pt2, color, thickness)


def draw_matches(
    img0,
    img1,
    kpts0,
    kpts1,
    match_indices,
    match_scores,
    max_display=5000,
    score_thresh=0.0,
    return_img=False,
):
    img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if img0.ndim == 2 else img0.copy()
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()

    h0, w0 = img0_color.shape[:2]
    h1, w1 = img1_color.shape[:2]
    out_img = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    out_img[:h0, :w0] = img0_color
    out_img[:h1, w0:] = img1_color

    # Shift keypoints in img1 for visualization
    kpts1_shifted = kpts1 + np.array([w0, 0], dtype=np.float32)

    # Gather good matches
    matches = [
        (i, match_indices[i], match_scores[i])
        for i in range(len(match_indices))
        if match_indices[i] >= 0 and match_scores[i] >= score_thresh
    ]
    matches = sorted(matches, key=lambda x: -x[2])[:max_display]

    for i, j, score in matches:
        pt0 = tuple(map(int, kpts0[i]))
        pt1 = tuple(map(int, kpts1_shifted[j]))
        color = (0, int(score * 255), 255 - int(score * 255))
        cv2.circle(out_img, pt0, 5, color, -1)
        cv2.circle(out_img, pt1, 5, color, -1)
        cv2.line(out_img, pt0, pt1, color, 3)

    if return_img:
        return out_img  # BGR
    else:
        from matplotlib import pyplot as plt

        plt.figure(figsize=(20, 10))
        plt.imshow(out_img[..., ::-1])
        plt.axis("off")
        plt.title("SuperGlue Matches")
        plt.show()


def draw_2d_3d_matches(
    image, qvec, tvec, matched_2d, matched_3d, K, dist_coeffs=None, return_img=False
):
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    draw_image = image.copy()

    # qvec should be xyzw
    rotation_matrix = R.from_quat(qvec).as_matrix()
    rvec, _ = cv2.Rodrigues(rotation_matrix)

    # draw points
    points_2d = project_points(matched_3d, rvec, tvec, K, dist_coeffs)
    draw_points(draw_image, points_2d, radius=6)
    draw_points(draw_image, matched_2d, radius=6, color=(255, 0, 0))

    # draw lines
    draw_lines(draw_image, matched_2d, points_2d)
    cv2.putText(
        draw_image,
        "#inlers:" + str(len(matched_2d)),
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 0, 255),
        2,
    )

    if return_img:
        return draw_image
    else:
        cv2.imshow("2d_3d_matches", draw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Project 3d points to image, if there is 2d keypoints, draw them as also
def project_3d_points_to_image(
    image, qvec, tvec, points_3d, K, dist_coeffs=None, keypoints_2d=None
):
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # qvec should be xyzw
    rotation_matrix = R.from_quat(qvec).as_matrix()
    rvec, _ = cv2.Rodrigues(rotation_matrix)

    points_2d = project_points(points_3d, rvec, tvec, K, dist_coeffs)
    draw_points(image, points_2d)

    if keypoints_2d is not None:
        print('Draw keypoints_2d in red on the image!')
        draw_points(image, keypoints_2d, color=(0, 0, 255))

    return image


def project_mesh_to_image(image, rotation, position, mesh_path, intrinsic, save_dir):
    # Read mesh once
    global MESH
    if MESH is None:
        MESH = o3d.io.read_triangle_mesh(mesh_path)
        MESH.compute_vertex_normals()

    height, width = image.shape[:2]
    intrinsic = intrinsic[0]

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = position.squeeze()

    # Use OffscreenRenderer(in docker window size will not be the same as what you set)
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    render.scene.set_background([0, 0, 0, 0])  # Transparent background

    # Add geometry
    if hasattr(o3d.visualization.rendering, 'MaterialRecord'):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        render.scene.add_geometry("mesh", MESH, mat)
    else:
        render.scene.add_geometry("mesh", MESH)

    # Set up camera
    render.setup_camera(intrinsic, extrinsic, width, height)

    rendered_img = np.asarray(render.render_to_image())  # RGB image

    # Prepare alpha mask(where mesh is drawn)
    alpha = 0.7
    gray = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2GRAY)
    alpha_mask = np.where(gray > 1, int(alpha * 255), 0).astype(np.uint8)

    # Prepare mesh overlay with alpha channel
    mesh_rgba = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2RGBA)
    mesh_rgba[..., 3] = alpha_mask

    # Background to RGB (if BGR)
    if image.shape[-1] == 3:
        background_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        background_rgb = image

    # Compose final image as RGBA
    composite_rgba = np.dstack(
        (background_rgb, np.full((height, width), 255, dtype=np.uint8))
    )  # Make background fully opaque
    overlay_rgba = mesh_rgba

    # Alpha blending manually
    alpha_f = overlay_rgba[..., 3:4].astype(np.float32) / 255.0
    composite_rgba[..., :3] = (1.0 - alpha_f) * composite_rgba[..., :3] + alpha_f * overlay_rgba[
        ..., :3
    ]
    composite_rgba = composite_rgba.astype(np.uint8)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "project_mesh_overlay.png")
    cv2.imwrite(output_path, cv2.cvtColor(composite_rgba, cv2.COLOR_RGBA2BGRA))

    debug_path = os.path.join(save_dir, "rendered_mesh_only.jpg")
    cv2.imwrite(debug_path, cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))


mesh_vertices_and_faces_ = None


def project_mesh_to_image_opencv(
    background_image,
    rotation_matrix,
    translation_vector,
    mesh_ply_path,
    camera_matrix,
    save_dir,
    line_thickness=1,
    border=30,
    min_depth=0.1,
):
    """
    Projects a mesh onto a background image, rendering its faces with transparency.

    Returns:
        result_image (np.ndarray): Background with projected mesh faces blended.
    """
    # Load mesh and extract vertices & faces
    global mesh_vertices_and_faces_  # Declare you want to use the global variable
    if mesh_vertices_and_faces_ is None:
        mesh = o3d.io.read_triangle_mesh(mesh_ply_path)
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.int32)
        mesh_vertices_and_faces_ = [vertices, faces]
    else:
        vertices = mesh_vertices_and_faces_[0]
        faces = mesh_vertices_and_faces_[1]

    dist_coeffs = np.zeros((4, 1))  # No distortion

    # Vertices in camera frame (for depth)
    vertices_cam = (rotation_matrix @ vertices.T + translation_vector.reshape(3, 1)).T  # (N,3)
    depths = vertices_cam[:, 2]  # Z
    min_d, max_d = min_depth, np.max(depths)
    depth_range = max_d - min_d if max_d > min_d else 1e-6

    # Project vertices
    projected_pts, _ = cv2.projectPoints(
        vertices_cam, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs
    )
    projected_pts = projected_pts.squeeze()  # shape (N, 2)

    # Prepare overlay for drawing
    h, w = background_image.shape[:2]

    # Draw edges of each face
    for tri in faces:
        # Draw triangle edges
        tri_depth = depths[tri]
        if np.any(tri_depth < min_depth):
            continue
        pts = projected_pts[tri].astype(np.int32)
        if (
            np.any(pts < -border)
            or np.any(pts[:, 0] >= w + border)
            or np.any(pts[:, 1] >= h + border)
        ):
            continue

        for i in range(3):
            pt1 = tuple(pts[i])
            pt2 = tuple(pts[(i + 1) % 3])
            avg_depth = (tri_depth[i] + tri_depth[(i + 1) % 3]) / 2
            # Normalize to [0, 1]
            t_norm = (avg_depth - min_d) / depth_range
            t_norm = np.clip(t_norm, 0, 1)

            # Color map: near = red, far = blue
            color = (int(255 * (1 - t_norm)), 0, int(255 * t_norm))  # B  # G  # R
            cv2.line(background_image, pt1, pt2, color, thickness=line_thickness)
    # cv2.imwrite(save_dir + "_project_mesh_overlay.png", background_image)
    return background_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', help='database name', default="database_3d.db", type=str)
    parser.add_argument('--images', help='images folder', default="images", type=str)
    parser.add_argument(
        '--model_path',
        help='model path',
        default="/mnt/ml-experiment-data/yeliu/gaussian_splatting/GoPro/NanshaOffice",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    db_path = os.path.join(args.model_path, args.database)

    map_drawer = MapDrawer(db_path)

    print('Main function')
