# TODO: include boolean option to perform offset augmentation or not

import os
from pathlib import Path
from PIL import Image

dataset_dir = (Path(__file__) / "../../cv_sample_data_yolo").resolve()

def extract_reid_data(dataset_split_path, output_path):
  labels_dir = dataset_split_path / "labels"

  for label_filename in labels_dir.glob("*.txt"):
    # assumes img is PNG
    img_path = dataset_split_path / "images" / (label_filename.stem + ".png")
    img = Image.open(img_path)
    
    with open(label_filename) as label_file:
      for line in label_file:
        if len(line.strip()) == 0:
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

        num_existing_imgs_of_obj = len(list(output_path.glob(f"{obj_id}_*.png")))
        img_cropped.save(output_path / f"{obj_id}_c1s1_{num_existing_imgs_of_obj + 1}_00.png")

reid_dir = dataset_dir / "reid"

try:
  os.mkdir(reid_dir)
  os.mkdir(reid_dir / "bounding_box_train")
  os.mkdir(reid_dir / "bounding_box_test")
  os.mkdir(reid_dir / "query")
except FileExistsError as err:
  pass

extract_reid_data(dataset_dir / "train", reid_dir / "bounding_box_train")
extract_reid_data(dataset_dir / "val", reid_dir / "bounding_box_test")
