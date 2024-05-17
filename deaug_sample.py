import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import multiprocessing
import torchvision.transforms.functional as TF

tensor_dir = './data/fake/'
save_dir = './samples/'
tensors = os.listdir(tensor_dir)[:1024]
process_num = 8

def denormalize_img(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0):
    if isinstance(img_tensor, torch.Tensor):
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)

        img = img_tensor.cpu().clone()  # 이미지 텐서를 복제하여 원본 텐서 보존
        img = img * std * max_pixel_value + mean * max_pixel_value
        img = torch.clamp(img, 0, max_pixel_value)  # 픽셀 값을 0에서 max_pixel_value 사이로 제한

        return img.byte()  # torch.ByteTensor로 변환하여 반환
    else:
        raise ValueError("Input image must be a torch.Tensor.")
    
def sampling(proc_id):
    
    for idx, a_tensor in enumerate(tensors):
        
        if idx < len(tensors) // process_num * proc_id:
            continue
        if idx >= len(tensors) // process_num * (proc_id + 1):
            break

        img = torch.load(tensor_dir + a_tensor)
        img = denormalize_img(img)
        img = TF.to_pil_image(img)
        img.save(save_dir + str(idx) + '.jpg')
            
        if idx % 50 == 0:
            print((idx - (len(tensors) // process_num * proc_id)) / (len(tensors) / process_num) * 100)


if __name__ == '__main__':

    procs = []
    for proc_id in range(process_num):
        p = multiprocessing.Process(target=sampling, args=(proc_id, ))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()