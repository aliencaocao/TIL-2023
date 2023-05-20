import os
import re
import shutil

test_dir = r'../RT-DETR/dataset/reid/bounding_box_test'
done_cls = []

for img in os.listdir(test_dir):
    cls = re.search(r'([-\d]+)_c(\d)', img).groups()[0]
    if cls not in done_cls:
        done_cls.append(cls)
        img_path = os.path.join(test_dir, img)
        shutil.copy(img_path, f'../RT-DETR/dataset/reid/query/{img}')
