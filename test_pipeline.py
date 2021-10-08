import time
import torch
import argparse
import cv2 as cv
import numpy as np
from skimage.transform import resize

from utils import autocrop, overlay_masks, display_masks
from models import detection_model, segmentation_model, get_preprocessing, get_validation_augmentation

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(img_path, augmentation, preprocessing):
    start_time = time.time()
    input = cv.imread(img_path)
    x, y, w, h = autocrop(input)
    cropped_input = input[y:y+h, x:x+w]
    auto_crop_time = time.time()
    print("--- Autocrop: %s seconds ---" % (auto_crop_time - start_time))
    detection_output = faster_rcnn(cropped_input)['instances']
    cropped_input = cropped_input[:,:,::-1] # bgr 2 rgb
    pred_classes = detection_output.pred_classes.cpu().numpy()
    pred_boxes = [x.cpu().numpy().astype(np.int) for x in detection_output.pred_boxes]
    detection_time = time.time()
    print("--- Detection: %s seconds ---" % (detection_time - auto_crop_time))
    pred_masks = []
    for e, cls in enumerate(pred_classes):
        xmin, ymin, xmax, ymax = pred_boxes[e]
        segm_input = preprocessing(image=augmentation(image = cropped_input[ymin:ymax, xmin:xmax])['image'])['image']
        mask = deeplabv3plus(torch.from_numpy(segm_input).unsqueeze(0).to(DEVICE).float())[0].detach().cpu().numpy()[cls]
        mask = resize(mask, (ymax-ymin, xmax-xmin))
        bg = np.zeros((h, w))
        if np.where(mask > 0.95, 1, 0).sum() > 0: # condition to include detection
            bg[ymin:ymax, xmin:xmax] = np.where(mask > 0.5, 1, 0)
        pred_masks.append(bg.astype(np.uint8))
    segm_time = time.time()
    print("--- Segmentation: %s seconds ---" % (segm_time - detection_time))
    print("--- Total: %s seconds ---" % (segm_time - start_time))

    if len(pred_masks) > 0: # show overlay masks if there is at least 1 detection
        print("Visualizing result...")
        overlayed = overlay_masks(cropped_input, pred_masks)
        display_masks(cropped_input, overlayed)
    else:
        print("No threats detected!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test pipeline for object separation framework.")
    parser.add_argument("--img_path", type=str, help="absolute path to test image")
    args = parser.parse_args()

    print("Loading models...")
    faster_rcnn = detection_model() # detection model
    deeplabv3plus, preprocessing_fn = segmentation_model() # segmentation model
    augmentation = get_validation_augmentation()
    preprocessing = get_preprocessing(preprocessing_fn)
    print("Models loaded!")

    print("Starting detection...")
    main(args.img_path, augmentation=augmentation, preprocessing=preprocessing)