import csv
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

preds_csv_path = Path("../RT-DETR/submission.csv")
test_imgs_path = Path("../RT-DETR/dataset/test/images")
output_path = Path("../RT-DETR/dataset/reid/test")


os.makedirs(output_path, exist_ok=True)

with open(preds_csv_path) as preds_file:
  reader = csv.reader(preds_file)
  next(reader) # skip header

  num_dets = {} # num_dets[i] is no. of bboxes in image i

  for row in tqdm(reader):
    filename = row[0]
    cam_id = int(filename.split("_")[1])
    y1, x1, y2, x2 = [float(x) for x in row[3:]]

    if cam_id not in num_dets:
      num_dets[cam_id] = 1
    else:
      num_dets[cam_id] += 1
    
    img = Image.open(test_imgs_path / (filename + ".png"))
    img_cropped = img.crop((
      x1 * img.width,
      y1 * img.height,
      x2 * img.width,
      y2 * img.height,
    ))

    img_cropped.save(output_path / f"-1_c{cam_id}s1_1_{num_dets[cam_id]}.png")
