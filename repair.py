from gfpgan import GFPGANer
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

restorer = None
restorer_version = ""

def repair(img, version="1.4", arch='clean', channel_multiplier=2, upscale=2, bg_upsampler=None):
    global restorer
    global restorer_version
    img = np.array(img)
    # ------------------------ set up GFPGAN restorer ------------------------
    if version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {version}.')

    model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url
    
    if restorer_version != version:
        restorer_version = version
        restorer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler)
        
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.5)

    return restored_img
    
if __name__ == '__main__':
    img_path = './images/ycy.png'
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    restored_img = repair(input_img)
    print(f'Processing {Path(img_path).name} ...')
    # cv2.imshow('y', restored_img)
    # cv2.waitKey(0)
    print(type(restored_img))