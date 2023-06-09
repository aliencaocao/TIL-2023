# 10000SGDMRT CV README

## Note on requirements.txt
We don't have one. This is because the PyTorch version needs to be downgraded so that mmcv-full can compile within a reasonable time. (If not, it has to be built from source, which takes up to 40 minutes.)

We do, however, have code cells in the Jupyter notebooks that will downgrade PyTorch and install all necessary packages.

## Instructions for running on Colab


## Pretrained models
Object detection: InternImage-L
- Repo link: https://github.com/OpenGVLab/InternImage/tree/master/detection
- Pretrained weights: https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_l_fpn_3x_coco.pth

ReID: SOLIDER-REID
- Repo link: https://github.com/tinyvision/SOLIDER-REID
- Pretrained weights: https://drive.google.com/file/d/1Y-RFAYdT56vnMjwxH1Ym3DVhZzZuMQZs/view?usp=share_link
