# TODO: include boolean option to perform offset augmentation or not

import os
from pathlib import Path

from PIL import Image
from shutil import copytree

dataset_dir = (Path(__file__) / "../../cv_sample_data_yolo").resolve()

def extract_reid_data(dataset_split_path, output_path):
  labels_dir = dataset_split_path / "labels"

  for img_idx, label_filename in enumerate(labels_dir.glob("*.txt")):
    # assumes img is PNG
    img_path = dataset_split_path / "images" / (label_filename.stem + ".png")
    img = Image.open(img_path)

    num_obj_instances = {}
    
    with open(label_filename) as label_file:
      for line in label_file:
        if line.strip() == "":
          continue

        line_split = line.split(" ")
        # not sure whether obj_id is 0- or 1-indexed; the +1 is there for safety
        obj_id = int(line_split[0].strip()) + 1
        x, y, w, h = [float(x.strip()) for x in line_split[1:]]

        x1 = x - w/2
        x2 = x + w/2
        y1 = y - h/2
        y2 = y + h/2

        img_cropped = img.crop((
          x1 * img.width,
          y1 * img.height,
          x2 * img.width,
          y2 * img.height,
        ))

        if obj_id not in num_obj_instances:
          num_obj_instances[obj_id] = 1
        else:
          num_obj_instances[obj_id] += 1

        img_cropped.save(output_path / f"{obj_id}_c{img_idx + 1}s1_1_{num_obj_instances[obj_id]}.png")

reid_dir = dataset_dir / "reid"
output_train_dir = reid_dir / "bounding_box_train"
output_test_dir = reid_dir / "bounding_box_test"
output_query_dir = reid_dir / "query"

try:
  os.mkdir(reid_dir)
  os.mkdir(output_train_dir)
  os.mkdir(output_test_dir)
except FileExistsError as err:
  pass

extract_reid_data(dataset_dir / "train", output_train_dir)
extract_reid_data(dataset_dir / "val", output_test_dir)
copytree(output_test_dir, output_query_dir)
