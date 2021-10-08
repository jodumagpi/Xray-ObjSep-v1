# Xray-ObjSep

![Pipeline Framework](https://github.com/jodumagpi/Xray-ObjSep/blob/main/pipeline_w_background.png)

This repository contains the evaluation pipeline code for our paper entitled "Pixel-level Analysis for Improving Threat Detection in X-ray Security Images". We also provide the pixel-level annotations on a randomly sampled subset from the [SIXray dataset](https://github.com/MeioJane/SIXray.git) as well as the list of mislabeled negative samples in the `data` folder.

This code was tested on an Ubuntu 18.04.5 machine with Python 3.7.11.  
Please install `detectron2`, `smp`, and `albumentations`.
```
# install detectron2
git clone https://github.com/facebookresearch/detectron2.git detectron2_repo
pip install -e detectron2_repo
# install smp and albumentations
pip install -U segmentation-models-pytorch albumentations --user
```

To run the code:
```
python test_pipeline.py --img_path /path/to/image.jpg
```

![Overlayed output mask](https://github.com/jodumagpi/Xray-ObjSep-v0.1/blob/a4f46445593b2d6353cf40d5b0146ecac7584c32/results.png)

Classification mAP on SIXray100 subset:
| Method | mean | Gun | Knife | Wrench | Pliers | Scissors |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| [CHR](https://arxiv.org/abs/1901.00303) | 64.54 | 98.60 | 81.09 | 46.69 | 62.72 | 33.61 |
| [GBAD](https://www.jstage.jst.go.jp/article/transinf/E103.D/2/E103.D_2019EDL8154/_article) | 66.14 | 95.74 | 85.25 | 54.87 | 53.10 | 41.74 |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870) | 86.55 | 98.74 | 77.72 | 78.35 | 89.78 | 77.11  |
| Proposed | 91.40 | 99.46 | 90.24 | 86.88 | 94.37 | 86.06 |


# References
* [Detectron2's Faster R-CNN Implementation](https://github.com/facebookresearch/detectron2.git)
* [SMP's DeepLabV3+ Implementation](https://github.com/qubvel/segmentation_models.pytorch.git)
