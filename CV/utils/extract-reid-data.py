# TODO: include boolean option to perform offset augmentation or not

import os
from pathlib import Path
from PIL import Image

dataset_dirs = [
  "../cv_sample_data_yolo/train",
  "../cv_sample_data_yolo/val",
  "../cv_sample_data_yolo/test",
]

def extract_reid_data(dataset_dir):
  dataset_path = Path(dataset_dir)
  reid_data_path = dataset_path / "reid"

  try:
    os.mkdir(reid_data_path)
  except FileExistsError:
    print(f"Folder {reid_data_path} already exists")

  labels_dir = dataset_path / "labels"

  for label_filename in labels_dir.glob("*.txt"):
    # assumes img is PNG
    img_path = dataset_path / "images" / (label_filename.stem + ".png")
    img = Image.open(img_path)
    
    with open(label_filename) as label_file:
      for line in label_file:
        line_split = line.split(" ")
        # not sure whether obj_id is 0- or 1-indexed; the +1 is there for safety
        obj_id = int(line_split[0]) + 1
        x, y, w, h = [float(x) for x in line_split[1:]]

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

        img_cropped.save(dataset_path / "reid" / f"{obj_id}_c1s1_000001_00.png")

for dataset_dir in dataset_dirs:
  extract_reid_data(dataset_dir)
