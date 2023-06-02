import argparse
import os
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# argparse code to accept the arguments --dataset and --output_dir
parser = argparse.ArgumentParser(description='ReID Inference')
parser.add_argument('--suspects_dir', default="", help='Path to suspects folder.', type=str)
parser.add_argument('--test_set_img_dir', default="", help='Path to test set images folder.', type=str)
parser.add_argument('--preds_path', default="", help='Path to preds csv file.', type=str)
parser.add_argument('--output_dir', default="", help='Path to output folder.', type=str)
args = parser.parse_args()

suspects_dir = Path(args.suspects_dir)
test_set_img_dir = Path(args.test_set_img_dir)
output_dir = Path(args.output_dir)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

preds_path = Path(args.preds_path)

# Read the preds bbox csv file
preds_df = pd.read_csv(preds_path)

for curr_img_id in tqdm(preds_df['Image_ID'].unique()):
    # read the suspect image
    suspect_img_path = suspects_dir/f"{curr_img_id}.png"
    suspect_img = Image.open(suspect_img_path)

    # read the corresponding test set image
    test_set_img_path = test_set_img_dir/f"{curr_img_id}.png"
    test_set_img = Image.open(test_set_img_path)

    # get the relevant slice of the preds_df
    relevant_slice = preds_df[preds_df['Image_ID'] == curr_img_id]

    # for each row in the relevant slice, draw the bbox on the suspect image
    for _, row in relevant_slice.iterrows():
        # annotate the test set image
        draw_on_test_set_img = ImageDraw.Draw(test_set_img)
        bbox = (
            row['xmin'] * test_set_img.width, 
            row['ymin'] * test_set_img.height, 
            row['xmax'] * test_set_img.width,
            row['ymax'] * test_set_img.height
        )
        if row['class']:
            # the current bbox we are drawing is a suspect bbox
            draw_on_test_set_img.rectangle(bbox, outline='#00ff00', width=2)
        else:
            # the current bbox we are drawing is a non-suspect bbox
            draw_on_test_set_img.rectangle(bbox, outline='#fa5f5f', width=2)

    # merge the suspect and test set image for final export
    output_width = suspect_img.width + test_set_img.width
    output_height = max(suspect_img.height, test_set_img.height)
    merged_image = Image.new('RGB', (output_width, output_height))
    # paste the suspect image at the top left corner
    merged_image.paste(suspect_img, (0, 0))
    # paste the test set image right next to the suspect image
    merged_image.paste(test_set_img, (suspect_img.width, 0))

    # annotate the merged image with the image filename
    draw_on_merged_img = ImageDraw.Draw(merged_image)
    text = f'{curr_img_id}.png'
    font = ImageFont.load_default()
    text_width, text_height = draw_on_merged_img.textsize(text, font=font)
    text_position = (0, suspect_img.height + 10)
    draw_on_merged_img.text(text_position, text, font=font, fill=(255, 255, 255))

    merged_image.save(output_dir/f'{curr_img_id}.png')