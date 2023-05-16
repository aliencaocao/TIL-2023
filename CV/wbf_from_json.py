from pathlib import Path
import glob
import json
from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion


def unnormalized_xywh_to_normalized_xyxy(unnormalized_xywh, img_width, img_height):
    x, y, w, h = unnormalized_xywh
    return [
        x / img_width,
        y / img_height,
        (x + w) / img_width,
        (y + h) / img_height,
    ]


def normalized_xyxy_to_unnormalized_xywh(normalized_xyxy, img_width, img_height):
    x1, y1, x2, y2 = normalized_xyxy
    return [
        x1 * img_width,
        y1 * img_height,
        (x2 - x1) * img_width,
        (y2 - y1) * img_height,
    ]


all_models_preds = {}
for preds_file_path in Path('to_wbf').glob("*.json"):
    with open(preds_file_path) as preds_file:
        preds = json.load(preds_file)

    all_models_preds[preds_file_path.stem] = {}

    for pred in preds:
        if pred["image_id"] not in all_models_preds[preds_file_path.stem]:
            all_models_preds[preds_file_path.stem][pred["image_id"]] = []
        all_models_preds[preds_file_path.stem][pred["image_id"]].append(pred)

wbf_all_preds = []
img_has_pred = {}
img_info = json.loads(open("data/finals/qualifiers_finals_no_annotations.json").read())["images"]
for img in img_info:
    boxes_list = []
    scores_list = []
    labels_list = []
    weights = [1] * len(all_models_preds)  # all models have same weightage

    for model in all_models_preds:
        this_model_preds = all_models_preds[model][img["id"]]
        boxes_list.append([unnormalized_xywh_to_normalized_xyxy(pred["bbox"], img["width"], img["height"]) for pred in this_model_preds])
        scores_list.append([pred["score"] for pred in this_model_preds])
        labels_list.append([pred["category_id"] for pred in this_model_preds])

    # log malformed boxes (several occur due to floating-point imprecision)
    # for i, model_boxes in enumerate(boxes_list):
    #   for j, box in enumerate(model_boxes):
    #     if box[0] < 0 or box[1] < 0 or box[2] > 1 or box[3] > 1:
    #       print(box, scores_list[i][j], labels_list[i][j])

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights)
    for box, score, label in zip(boxes, scores, labels):
        wbf_all_preds.append({
            "image_id": img["id"],
            "bbox": normalized_xyxy_to_unnormalized_xywh(box, img["width"], img["height"]),
            "category_id": label,
            "score": score,
        })

    img_has_pred[img["id"]] = len(boxes) > 0

for img_id in img_has_pred:
    if not img_has_pred[img_id]:
        wbf_all_preds.append({
            "image_id": img_id,
            "bbox": [0., 0., 0., 0.],
            "category_id": 2,
            "score": 0.0000001,
        })

try:
    with open(f"wbf_({', '.join(all_models_preds)}).json", "w") as wbf_preds_file:
        json.dump(wbf_all_preds, wbf_preds_file, separators=(",", ":"))
except OSError:
    with open(f"wbf_error_please_rename_manually.json", "w") as wbf_preds_file:
        json.dump(wbf_all_preds, wbf_preds_file, separators=(",", ":"))
