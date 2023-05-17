# relabels all YOLO txt annotations to have class 0
# because all plushies are of the same class

import os
import csv
from pathlib import Path

txt_dirs = [
  "../cv_sample_data_yolo/train/labels",
  "../cv_sample_data_yolo/val/labels",
  
  # obviously there will be no labels for test set so remove this
  "../cv_sample_data_yolo/test/labels",
]

def relabel_dir(txt_dir):
  relabelled_dir = Path(txt_dir).parent / "relabelled"
  
  try:
    os.mkdir(relabelled_dir)
  except FileExistsError:
    print(f"Folder {relabelled_dir} already exists")

  for ann_file_path in Path(txt_dir).glob("*.txt"):
    relabelled_file_path = relabelled_dir / ann_file_path.name

    with open(ann_file_path) as infile, open(relabelled_file_path, "w") as outfile:
      reader = csv.reader(infile, delimiter=" ")
      writer = csv.writer(outfile, delimiter=" ")
      for row in reader:
        writer.writerow([0] + row[1:])

for txt_dir in txt_dirs:
  relabel_dir(txt_dir)
