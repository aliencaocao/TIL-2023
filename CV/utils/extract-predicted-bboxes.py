import csv
import json
import os
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

pred_path = "../InternImage/submissions/epoch 12 conf 0.99.csv"
preds_csv_path = Path(pred_path)
test_imgs_path = Path("../RT-DETR/dataset/test/images/")
output_path = Path("../RT-DETR/dataset/reid/test_old/")
suspect_imgs_path = Path("../RT-DETR/dataset/reid/suspects/")

os.makedirs(output_path, exist_ok=True)

mapping_dict = {}

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

    gallery_dir = output_path / "bounding_box_test"
    query_dir = output_path / "query"
    os.makedirs(gallery_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)

    img_save_path = gallery_dir / f"-1_c{cam_id}s1_1_{num_dets[cam_id]}.png"
    img_cropped.save(img_save_path)
    shutil.copy(suspect_imgs_path / (filename + ".png"), query_dir)

    mapping_dict[str(img_save_path)] = {
      'confidence': row[2],
      'ymin': y1,
      'xmin': x1,
      'ymax': y2,
      'xmax': x2,
    }

with open(f'{pred_path.split(".csv")[0]}_bbox_mapping.json', 'w') as f:
  json.dump(mapping_dict, f)
