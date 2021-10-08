import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def detection_model():
    """
       The configuration of our trained model can be found in the models/configs folder.
       To change the pretrained weights, please modify the config file.
    """
    cfg = get_cfg()
    cfg.merge_from_file("models/configs/faster_rcnn_091321.yaml")
    model = DefaultPredictor(cfg) # alreay on GPU and in eval mode

    return model