import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import multiprocessing

image_dir1 = '../CoAtNet/pytorch-image-models/data/train/maximum-speed-limit-50/'
image_dir2 = '../CoAtNet/pytorch-image-models/data/train/others/'
save_dir1 = './data/fake/'
save_dir2 = './data/real/'
images1 = os.listdir(image_dir1)[:512]
images1 = [image_dir1 + an_image for an_image in images1]
images2 = os.listdir(image_dir2)[:14114]
images2 = [image_dir2 + an_image for an_image in images2]
images = images1 + images2
org_tensor_dir = './original_tensor/'
tensors = os.listdir(org_tensor_dir)
process_num = 8


def hsv2colorjitter(h, s, v):
    """Map HSV (hue, saturation, value) jitter into ColorJitter values (brightness, contrast, saturation, hue)"""
    return v, v, s, h

# config : Changed setting for the xlarge_0325 experiments (1) 
def classify_albumentations(
        augment=True,
        size=224,
        aug_params={}
):
    """YOLOv8 classification Albumentations (optional, only used if package is installed)."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        crop_rate = aug_params['crop_rate']
        rotate = aug_params['rotate']
        p_rotate = aug_params['p_rotate']
        shear = aug_params['shear']
        p_shear = aug_params['p_shear']
        perspective = aug_params['perspective']
        p_perspective = aug_params['p_perspective']
        hsv_h = aug_params['hsv_h']
        hsv_s = aug_params['hsv_s']
        hsv_v = aug_params['hsv_v']
        p_hsv = aug_params['p_hsv']
        motion_blur_limit = aug_params['motion_blur_limit']
        p_motion_blur = aug_params['p_motion_blur']
        gaussian_blur_limit = aug_params['gaussian_blur_limit']
        p_gaussian_blur = aug_params['p_gaussian_blur']
        gaussian_noise_var_limit = aug_params['gaussian_noise_var_limit']
        p_gaussian_noise = aug_params['p_gaussian_noise']

        if augment:
            T = []
            T += [A.RandomCropFromBorders(crop_left=crop_rate, crop_right=crop_rate, crop_top=crop_rate, crop_bottom=crop_rate)]
            T += [A.LongestMaxSize(max_size=size)]

            
            if gaussian_noise_var_limit and p_gaussian_noise > 0:
                assert len(gaussian_noise_var_limit) == 2, 'gaussian_noise_var_limit must be [min, max] in either tuple or list'
                T += [A.GaussNoise(var_limit=gaussian_noise_var_limit, p=p_gaussian_noise)]

            if any((hsv_h, hsv_s, hsv_v)) and p_hsv > 0:
                T += [A.ColorJitter(*hsv2colorjitter(hsv_h, hsv_s, hsv_v), p=p_hsv)]  # brightness, contrast, saturation, hue
            if rotate > 0:
                T += [A.Rotate(limit=rotate, p=p_rotate)]
            if shear > 0:
                T += [A.Affine(shear=(-shear, shear), p=p_shear, mode=cv2.BORDER_REFLECT)]
            if perspective > 0:
                T += [A.Perspective(scale=perspective, p=p_perspective)]

            if gaussian_blur_limit and p_gaussian_blur > 0:
                assert len(gaussian_blur_limit) == 2, 'gaussian_blur_limit must be [min, max] in either tuple or list'
                T += [A.GaussianBlur(blur_limit=gaussian_blur_limit, p=p_gaussian_blur)]

            if motion_blur_limit and p_motion_blur > 0:
                assert len(motion_blur_limit) == 2, 'motion_blur_limit must be [min, max] in either tuple or list'
                T += [A.MotionBlur(blur_limit=motion_blur_limit, p=p_motion_blur)]

        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.LongestMaxSize(max_size=size)]
        T += [A.Normalize()]  # Normalize and convert to Tensor
        T += [A.PadIfNeeded(min_height=size, min_width=size, position=A.PadIfNeeded.PositionType.TOP_LEFT, border_mode=cv2.BORDER_CONSTANT, value=0)]
        T += [ToTensorV2()]
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        print('package not installed!!!')
        pass


def aug_move(proc_id, aug_params):
    
    augment = classify_albumentations(augment=True, aug_params=aug_params)
    
    for idx, an_image in enumerate(images):
        
        if idx < len(images) // process_num * proc_id:
            continue
        if idx >= len(images) // process_num * (proc_id + 1):
            break

        org_img = Image.open(an_image)
        org_img = np.asarray(org_img)
        aug_img = augment(image=org_img)['image']
        torch.save(aug_img, save_dir1 + str(idx) + '.pt')
        
        if idx % 100 == 0:
            print((idx - (len(images) // process_num * proc_id)) / (len(images) / process_num) * 100)


def move(proc_id):
    for idx, a_tensor in enumerate(tensors):
        if idx < len(tensors) // process_num * proc_id:
            continue
        if idx >= len(tensors) // process_num * (proc_id + 1):
            break
        loaded_ts = torch.load(org_tensor_dir + a_tensor)
        torch.save(loaded_ts, save_dir2 + str(idx) + '.pt')
        
        if idx % 100 == 0:
            print((idx - (len(tensors) // process_num * proc_id)) / (len(tensors) / process_num) * 100)


def prepare_dataset(aug_params):
    procs = []
    for proc_id in range(process_num):
        p = multiprocessing.Process(target=aug_move, args=(proc_id, aug_params))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    
    print("--------------- aug_move done! ---------------")
        
if __name__ == '__main__':
    aug_params = {
        'crop_rate' : (0.4 / 1.4),
        'rotate':15,
        'p_rotate':0.7,
        'shear':10,
        'p_shear':0.5,
        'perspective':0.1,
        'p_perspective':0.5,
        'hsv_h':0.05,
        'hsv_s':0.7,
        'hsv_v':0.6,
        'p_hsv':0.5,
        'motion_blur_limit':[7, 51],
        'p_motion_blur':0.7,
        'gaussian_blur_limit':[9, 55],
        'p_gaussian_blur':0.95,
        'gaussian_noise_var_limit':[10.0, 90.0],
        'p_gaussian_noise':0.25
    }
    prepare_dataset(aug_params)

    
