import json

def add_segmentation(ann_file):
    with open(ann_file) as f:
        json_data = json.load(f)
    
    for ann in json_data["annotations"]:
        # list of polygons in [x1, y1, x2, y2, ..., xn, yn] format
        x, y, w, h = ann["bbox"]
        ann["segmentation"] = [[
            x, y, # top left
            x+w, y, # top right
            x+w, y+h, # bottom right
            x, y+h, # bottom left
        ]]

    return json_data

ann_file_paths = [
    "../RT-DETR/dataset/train/train.json",
    "../RT-DETR/dataset/val/val.json",
]

for ann_file in ann_file_paths:
    new_json_data = add_segmentation(ann_file)
    with open(f"{ann_file[:-5]}_with_seg.json", "w") as f:
        json.dump(new_json_data, f)
