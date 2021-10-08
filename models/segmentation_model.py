import torch
import numpy as np
import albumentations as albu
import segmentation_models_pytorch as smp

def get_training_augmentation():
    train_transform = [
        albu.VerticalFlip(),
        albu.HorizontalFlip(),
        albu.Transpose(),
        #albu.Affine(translate_percent=(0, 0.25), mode=1),
        #albu.SafeRotate(180),
        albu.Resize(192, 192)
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(192, 192)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def segmentation_model():
    
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=5, 
        activation=ACTIVATION,)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # load the pretrained model here
    model.load_state_dict(torch.load("/content/deeplabv3+_092921.pth").state_dict())
    model = model.to(DEVICE)
    model = model.eval()

    return model, preprocessing_fn