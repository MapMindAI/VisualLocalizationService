import sqlite3
import numpy as np
import argparse
import os
import cv2
import matplotlib.pyplot as plt
from make_retrieval_db import BowRetireval


def plot_retrievals(query_image, retrieval_images):
    fig, axes = plt.subplots(2, len(retrieval_images), figsize=(15, 5))  # Adjust figsize as needed
    # Plot image_A in the first row
    axes[0, 0].imshow(query_image)
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Image Query")

    # Hide the remaining subplots in the first row
    for i in range(1, len(retrieval_images)):
        axes[0, i].axis("off")

    # Plot images_B in the second row
    for i, ax in enumerate(axes[1, :]):
        ax.imshow(retrieval_images[i])
        ax.axis("off")
        ax.set_title(f"Image {i+1}")

    # Show the plot
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", help="database name", default="database_3d.db", type=str)
    parser.add_argument("--images", help="images folder", default="images", type=str)
    parser.add_argument("--model_path", help="model path", default="", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    db_path = os.path.join(args.model_path, args.database)
    sqlite_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sqlite_cursor = sqlite_conn.cursor()

    def get_image(image_id):
        sqlite_cursor.execute("SELECT * FROM images WHERE image_id = ?", (str(image_id),))
        result = sqlite_cursor.fetchone()
        assert result is not None
        image_path = os.path.join(args.model_path, os.path.join(args.images, result[1]))
        print(f'id: {image_id}, path: {image_path}')
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (256, 256))
        return cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    retrieval = BowRetireval(args.model_path)

    import sys

    sys.path.append("python/visual_pose_service/feature")
    from superpoint import SuperPoint

    superpoint = SuperPoint("192.168.19.150:8001")

    test_image = cv2.imread(os.path.join(args.model_path, "test/1506876249785000.jpg"))
    ktps, descps, _ = superpoint.run(test_image)

    for i in range(ktps[0].shape[0]):
        pt0 = tuple(map(int, ktps[0][i]))
        cv2.circle(test_image, pt0, 3, (0, 255, 0), -1)

    retrievaled_ids = retrieval.retrieve_bow(descps[0])
    retrievaled_images = []
    for i in retrievaled_ids:
        retrievaled_images.append(get_image(i))

    plot_retrievals(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB), retrievaled_images)
