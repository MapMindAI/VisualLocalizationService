import sqlite3
import numpy as np
import cv2
import argparse
import os
import glob
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

import joblib
import faiss
from tqdm import tqdm

RETRIEVAL_BOW_NUM_CLUSTERS = 1000
RETRIEVAL_BOW_KMEANS_PATH = "retrieval_vocab_kmeans.pkl"
RETRIEVAL_BOW_IDS_PATH = "retrieval_ids.npy"
RETRIEVAL_BOW_VECTORS_PATH = "retrieval_bow_vectors.npy"


def compute_word(kmeans, desc):
    word_ids = kmeans.predict(desc)
    hist, _ = np.histogram(word_ids, bins=np.arange(RETRIEVAL_BOW_NUM_CLUSTERS + 1))
    hist = hist.astype("float32") / np.linalg.norm(hist)
    return hist


def quat_angle_diff(q1, q2):
    """
    Compute angular difference (in radians) between two unit quaternions
    using relative rotation matrix.
    """
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)

    # Relative rotation matrix
    r_rel = r1.inv() * r2
    R_rel = r_rel.as_matrix()

    # Clamp trace to avoid NaNs due to numerical error
    trace = np.trace(R_rel)
    trace = np.clip(trace, -1.0, 3.0)

    angle = np.arccos((trace - 1) / 2.0)  # in radians
    return angle


class BowRetireval:
    def __init__(self, model_path, top_k=5, db_image_poses=None):
        self.top_k = top_k
        self.kmeans = joblib.load(os.path.join(model_path, RETRIEVAL_BOW_KMEANS_PATH))
        self.image_ids = np.load(
            os.path.join(model_path, RETRIEVAL_BOW_IDS_PATH), allow_pickle=True
        )

        image_bow_vectors = np.load(os.path.join(model_path, RETRIEVAL_BOW_VECTORS_PATH))
        self.index = faiss.IndexFlatL2(image_bow_vectors.shape[1])
        self.index.add(image_bow_vectors)

        self.db_image_poses = db_image_poses
        self.kd_tree_top_k = top_k * 3
        self.create_kdtree()

    def create_kdtree(self):
        if (len(self.db_image_poses) < self.kd_tree_top_k):
            self.kd_tree = None
            return

        self.kd_tree_keys = []
        kd_points = []
        for key in self.db_image_poses:
            self.kd_tree_keys.append(key)
            kd_points.append(self.db_image_poses[key][1])
        self.kd_tree = cKDTree(np.array(kd_points))

    def retrieve_bow(self, desc):
        word_ids = self.kmeans.predict(desc)
        hist, _ = np.histogram(word_ids, bins=np.arange(RETRIEVAL_BOW_NUM_CLUSTERS + 1))
        hist = hist.astype("float32") / np.linalg.norm(hist)
        D, I = self.index.search(hist.reshape(1, -1), self.top_k)
        return [self.image_ids[i] for i in I[0]]

    # prior_pose = np.array([rot.w, rot.x, rot.y, rot.z, trans.x, trans.y, trans.z])
    def check_prior_pose_valid(self, pose):
        return pose[0] != 0.0;

    def retrieve_nearest_pose(self, pose):
        if self.kd_tree is None:
            return []
        # get closest poses in position
        _, indices = self.kd_tree.query(pose[4:], k=self.kd_tree_top_k)

        # compute rotation then get closest
        angle_diffs = []
        for id in indices:
            quad_ref = self.db_image_poses[self.kd_tree_keys[id]][0][0:4]
            angle_diffs.append(quat_angle_diff(quad_ref, pose[0:4]))

        top_local_indices = np.argsort(angle_diffs)[:self.top_k]
        top_indices = indices[top_local_indices]
        return [self.kd_tree_keys[i] for i in top_indices]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", help="database name", default="database_3d.db", type=str)
    parser.add_argument("--model_path", help="model path", default="", type=str)
    parser.add_argument("--feature_dim", help="feature dimension", default=256, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    db_path = os.path.join(args.model_path, args.database)

    sqlite_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sqlite_cursor = sqlite_conn.cursor()

    def get_descroptors(image_id):
        #  (image_id, rows, cols, data)
        sqlite_cursor.execute("SELECT * FROM descriptors WHERE image_id = ?", (str(image_id),))
        result = sqlite_cursor.fetchone()
        assert result is not None
        desc = np.frombuffer(result[3], dtype=np.uint8).reshape((result[1], result[2]))
        return desc

    # Execute a SELECT query to get all rows from the table
    sqlite_cursor.execute(f"SELECT * FROM images;")
    rows = sqlite_cursor.fetchall()
    # Fetch all the rows
    all_descriptors = []
    progress_bar = tqdm(range(0, len(rows)), desc="Process Fetch All Descriptors")

    for row in rows:
        all_descriptors.append(get_descroptors(row[0]))
        progress_bar.update(1)
    progress_bar.close()

    all_descriptors = np.vstack(all_descriptors)  # shape: (N_total_kp, 256)
    print(" - ", all_descriptors.shape)

    print("Make Index")
    kmeans = MiniBatchKMeans(
        n_clusters=RETRIEVAL_BOW_NUM_CLUSTERS, batch_size=10000, verbose=1, n_init=3
    )
    kmeans.fit(all_descriptors)
    joblib.dump(kmeans, os.path.join(args.model_path, RETRIEVAL_BOW_KMEANS_PATH))

    progress_bar = tqdm(range(0, len(rows)), desc="Process Global Feature")

    image_ids = []
    image_bow_vectors = []
    for row in rows:
        word = compute_word(kmeans, get_descroptors(row[0]))
        image_bow_vectors.append(word)
        image_ids.append(row[0])
        progress_bar.update(1)

    progress_bar.close()
    np.save(os.path.join(args.model_path, RETRIEVAL_BOW_IDS_PATH), image_ids)
    np.save(os.path.join(args.model_path, RETRIEVAL_BOW_VECTORS_PATH), image_bow_vectors)

    print("Done!")
    sqlite_cursor.close()
    sqlite_conn.close()
