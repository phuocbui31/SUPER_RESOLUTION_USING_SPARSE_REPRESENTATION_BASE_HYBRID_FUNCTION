import numpy as np
import os
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from PIL import Image

from skimage import io as skio 
from skimage import color as skco
from skimage import transform

import warnings 

# import skimage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
import skimage.color as skcolor
# import ipdb

##################################################################################
#
#  image read/write operation
#
##################################################################################
def img_route(dataset_route = "./Set14", img_id = "001", method = "LR", scale = 2):
    # dataset_route = "/home/e/Eulring/GitProject/Mixture-Function-Sparse-Representation/dataset/Set14"
    # dataset_route = "/Users/eulring/Efile/Dataset/Set14"
    dataset_route = dataset_route
    print(f'img_route: {dataset_route}')
    
    img_route = os.path.join(dataset_route, "image_SRF_"+str(scale))
    img_name = '/img_' + img_id + '_SRF_' + str(scale) + '_' + method + '.png'
    print(img_route + img_name)
    return img_route + img_name



#=================================================================================
#  turn the img_route to the normalized numpy array of y channel directly 
def numpy_from_img_route(img_route, imagecut = False, batch_size = (6, 6)):
    img_rgb = Image.open(img_route)
    # img_ycbcr = rgb2ycbcr(img_rgb)
    img_ycbcr = skco.rgb2ycbcr(np.array(img_rgb))

    img1 = np.array(img_ycbcr)
    y = img1[:, :, 0]
    
    if imagecut == True:
        y = imgcut(y, batch_size = batch_size)
    
    if np.max(y) > 1: y = y / 255.0
        
    return y

# Tự thêm
def imgcut(img, batch_size=(6, 6)):
    h, w = img.shape[:2]
    bh, bw = batch_size
    
    patch_h = h // bh
    patch_w = w // bw
    
    patches = []
    for i in range(bh):
        for j in range(bw):
            patch = img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            patches.append(patch)
    
    return np.array(patches)



##################################################################################
#
#  Evalution
#
##################################################################################

#=================================================================================
# RMSE
def RMSE(img1, img2):
    if len(img1.shape) == 3:
        img1_y = skco.rgb2ycbcr(img1)[:, :, 0]
    else: 
        img1_y = img1
    if len(img2.shape) == 3:
        img2_y = skco.rgb2ycbcr(img2)[:, :, 0]
    else: 
        img2_y = img2

    img_dif = img1_y - img2_y
    
    rmse = np.sqrt(np.mean(img_dif**2))
    return rmse

 
def PSNR(img1, img2):
    rmse = RMSE(img1, img2)
    return 20 * np.log10(255.0/rmse)

def quick_eval(dataset_route, img_O, img_id = '001', scale = 2, file_name=None):
    # ipdb.set_trace()
    # img_B = skio.imread(img_route(img_id = img_id, method='bicubic', scale = scale))
    # img_H = skio.imread(img_route(dataset_route, img_id = img_id, method='HR', scale = scale))
    print(f'quick_eval: {dataset_route}')
    # if len(img_H.shape) == 2:
    #     img_H = skcolor.gray2rgb(img_H)
    
    # img_H, img_O = equal_size(img_H, img_O)

    # img_H, img_B = equal_size(img_H, img_B)
    # ipdb.set_trace()
    PSNR_1 = PSNR(img_H, img_O)
    # PSNR_2 = PSNR(img_H, img_B)
    
    SSIM_1 = SSIM(img_H, img_O)
    # SSIM_2 = SSIM(img_H, img_B)
    
    # print('PSNR/SSIM for Bicubic Interpolation: %f dB', PSNR_2, SSIM_2);
    # print('PSNR/SSIM for Sparse Representation Recovery: %f dB', PSNR_1, SSIM_1);

    # print(f'PSNR/SSIM for Bicubic Interpolation: {PSNR_2:.4f} dB / {SSIM_2:.4f}')
    print(f'PSNR/SSIM for {file_name}: {PSNR_1:.4f} dB / {SSIM_1:.4f}')
    return PSNR_1, SSIM_1

    
#=================================================================================
# PSNR
# refer : https://github.com/aizvorski/video-quality/blob/master/psnr.py
def PSNR1(img1, img2):
    import math
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def PSNR0(img1, img2):
    # from skimage.measure import compare_psnr
    # return compare_psnr(img1, img2, 255)
    return peak_signal_noise_ratio(img1, img2, data_range=255)

def PSNR2(img1, img2):
    diff = img1 - img2
    mse = np.mean(np.square(diff))
    psnr = 10.0 * np.log10(255.0 * 255 / mse)
    return psnr



#=================================================================================
# SSIM
# refer : https://github.com/aizvorski/video-quality/blob/master/ssim.py
def SSIM(img1, img2, data_range = 1.0, multichannel=False):
    img1_ = img1.copy()
    img2_ = img2.copy()
    if np.max(img1_) < 2.0 : img1_*=255.0
    if np.max(img2_) < 2.0 : img2_*=255.0
    data_range = 255.0
    if len(img1_.shape) == 3 : multichannel=True
    
    # 2
    return ssim(img1_, img2_, data_range=255.0, channel_axis=-1 if multichannel else None)

# 1


##################################################################################
#
#  Other
#
##################################################################################


def equal_size(img1, img2):
    w1, h1 = img1.shape[0], img1.shape[1]
    w2, h2 = img2.shape[0], img2.shape[1]
    
    w = min(w1, w2)
    h = min(h1, h2)
    
    if len(img1.shape) == 3:
        img1 = img1[0 : w, 0 : h, :]
        img2 = img2[0 : w, 0 : h, :]
    else:
        img1 = img1[0 : w, 0 : h]
        img2 = img2[0 : w, 0 : h]
    
    return img1, img2