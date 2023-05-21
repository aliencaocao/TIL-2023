from pathlib import Path
import os
import csv
from tqdm import tqdm
from PIL import Image

preds_csv_path = Path("../RT-DETR/submission.csv")
test_imgs_path = Path("/home/yip/Downloads/Test/images")
output_path = Path("../cv_sample_data_yolo/reid/obj_det_output_bboxes")

try:
  os.mkdir(output_path)
except FileExistsError:
  print(f"Folder {output_path} already exists.")

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
