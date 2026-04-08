import cv2
import sqlite3
import numpy as np
import sys
import argparse
import os
import json

from transform_colmap_model_qrcode import estimate_qrcode_poses_optimize


def adjust_map():
    print('hahha')

def compute_sim3_transform(src_pts, dst_pts):
    """Umeyama Sim(3): align src_pts to dst_pts."""
    assert len(src_pts) == len(dst_pts)
    src_pts = np.array(src_pts, dtype=np.float64)
    dst_pts = np.array(dst_pts, dtype=np.float64)

    # Centroids
    src_centroid = np.mean(src_pts, axis=0)
    dst_centroid = np.mean(dst_pts, axis=0)

    # Center
    src_centered = src_pts - src_centroid
    dst_centered = dst_pts - dst_centroid

    # Covariance
    H = src_centered.T @ dst_centered / len(src_pts)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Fix rotation so det(R) = +1
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.eye(3)
    D[2, 2] = d
    R = Vt.T @ D @ U.T

    # Scale
    var_src = np.sum(src_centered ** 2) / len(src_pts)
    scale = np.sum(S * np.diag(D)) / var_src

    # Translation
    t = dst_centroid - scale * R @ src_centroid

    return R, t, scale
  

def process_with_prior(original_db_path, new_db_path):
    # Load poses of qrcodes
    original_qr_poses = poses_from_json(os.path.join(original_db_path, 'qr_poses.json'))
    new_qr_poses = poses_from_json(os.path.join(new_db_path, 'qr_poses.json'))
    
    common_qr_ids = set(original_qr_poses.keys()) & set(new_qr_poses.keys())
    if not common_qr_ids:
        print("No common qr ids! Return!")
        return
      
    # Calculate sim3 transform
    src_pts = []
    dst_pts = []
    for qr_id in common_qr_ids:
        src_pts.append(new_qr_poses[qr_id]['t'])
        dst_pts.append(original_qr_poses[qr_id]['t'])
    R, t, scale = compute_sim3_transform(src_pts, dst_pts)
    print(f"R={R}, t={t}, scale={scale}")
    
    src_pts_transformed = (scale * (R @ np.array(src_pts).T)).T + t
    err = np.linalg.norm(src_pts_transformed - np.array(dst_pts), axis=1)
    print("Mean error:", np.mean(err))
    
    print('src_pts: ')
    print(src_pts)
    print('dst_pts: ')
    print(dst_pts)


def process(original_db_path, new_db_path, input_sparse_path, image_dir, qrcode_size, qrcode_distance_m):
    # 1. Load maps and estimate QR poses
    original_output = estimate_qrcode_poses_optimize(original_db_path, input_sparse_path, image_dir, qrcode_size, qrcode_distance_m)
    new_output = estimate_qrcode_poses_optimize(new_db_path, input_sparse_path, image_dir, qrcode_size, qrcode_distance_m)
    if original_output['qr_poses']['anchor_num'] == 0 or new_output['qr_poses']['anchor_num'] == 0:
        print("Detect 0 anchor in original/new map! Return!")
        return 
    common_qr_ids = set(original_output['qr_poses']['anchors'].keys()) & set(new_output['qr_poses']['anchors'].keys())
    if not common_qr_ids:
        print("No common qr ids! Return!")

    # 2. transform and scale from new to original
    src_pts = []
    dst_pts = []
    for qr_id in common_qr_ids:
        src_pts.append(new_output['qr_poses']['anchors'][qr_id]['t'])
        dst_pts.append(original_output['qr_poses']['anchors'][qr_id]['t'])
        
    R, t, scale = compute_sim3_transform(src_pts, dst_pts)
    print(f"R={R}, t={t}, scale={scale}")
    
    src_pts_transformed = (scale * (R @ np.array(src_pts).T)).T + t
    err = np.linalg.norm(src_pts_transformed - np.array(dst_pts), axis=1)
    print("Mean error:", np.mean(err))
    
    print('src_pts: ')
    print(src_pts)
    print('dst_pts: ')
    print(dst_pts)
     
    
def poses_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    poses = {}
    for qr_id, pose in data.items():
        poses[qr_id] = {
            'R': np.array(pose['R'], dtype=float),
            't': np.array(pose['t'], dtype=float)
        }
    
    return poses
    

def main():
    parser = argparse.ArgumentParser(description='Marker Processor')
    parser.add_argument(
        '--original_db_path',
        type=str,
        # required=True,
        default='/mnt/ml-experiment-data/yeliu/gaussian_splatting/GoPro/qrcode_test/room1_test/',
        help='Path to the original COLMAP model directory',
    )
    parser.add_argument('--new_db_path',
        type=str,
        default='/mnt/ml-experiment-data/yeliu/gaussian_splatting/GoPro/qrcode_test/room2_test/',
        help='Path to the new COLMAP model directory')
    parser.add_argument('--image_dir', default='images', help='Image dir')
    parser.add_argument('--qrcode_size', type=float, default=0.174, help='Size of the marker (m)')
    parser.add_argument('--input_sparse_path', default='sparse/0/', help='input sparse model path under colmap_model_dir')
    args = parser.parse_args()
    
    qr_distances_m = {
        ('mobili_vlp_anchor_02', 'mobili_vlp_anchor_03'): 2.3
    }
    # process(args.original_db_path, args.new_db_path, args.input_sparse_path,
    #         args.image_dir, args.qrcode_size, qr_distances_m)

    process_with_prior(args.original_db_path, args.new_db_path)


if __name__ == "__main__":
  main()