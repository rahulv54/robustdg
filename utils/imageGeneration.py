import numpy as np
import cv2
from skimage.draw import circle, disk, rectangle
from skimage import data, img_as_float
from skimage.util import random_noise
from matplotlib import pyplot as plt
import os

def generate_circle_img(sigma_range, center_extent, radius_extent, amount = 0, mode = 'gaussian', debug = False):
    center_coords = np.random.randint(*center_extent,2)
    radius = np.random.randint(radius_extent)
    

    rr, cc = disk((center_coords[0], center_coords[1]), radius)
    

    img = np.zeros((256,256), dtype=np.float32)
    mask = np.zeros((256,256), dtype=np.uint8)
    
    sigma = np.random.uniform(*sigma_range )
    amount = amount
    img = img_as_float(img)
    
    if mode == 'gaussian':
        noise = random_noise(img, var= sigma, mode = mode)
    else:
        noise = random_noise(img, amount= 0.5, mode = mode)
    
    img[rr, cc] = noise[rr, cc]
    mask[rr, cc] = 1

    if debug == True:
        plt.subplot(1,2,1)
        plt.imshow(img)

        plt.subplot(1,2,2)
        plt.imshow(mask)
        
    return img, mask

def generate_rect_img(sigma_range, start_extent, extent, amount = 0, mode = 'gaussian', debug = False):
    start_coords = np.random.randint(*start_extent, 2)
    extent = np.random.randint(extent)
    
    rr, cc = rectangle((start_coords[0], start_coords[1]), extent)
    
    img = np.zeros((256,256), dtype=np.float32)
    mask = np.zeros((256,256), dtype=np.uint8)
    
    sigma = np.random.uniform(*sigma_range )
    amount = amount
    img = img_as_float(img)
    
    if mode == 'gaussian':
        noise = random_noise(img, var= sigma, mode = mode)
    else:
        noise = random_noise(img, amount= 0.5, mode = mode)
        
    img[rr, cc] = noise[rr, cc]
    mask[rr, cc] = 1

    if debug == True:
        plt.subplot(1,2,1)
        plt.imshow(img)

        plt.subplot(1,2,2)
        plt.imshow(mask)
        
    return img, mask

def generate_colored_rect_img(sigma_range, start_extent, extent, color_index = 2, debug = False):
    start_coords = np.random.randint(*start_extent, 2)
    extent = np.random.randint(extent)
    
    rr, cc = rectangle((start_coords[0], start_coords[1]), extent)
    
    img = np.zeros((3,256,256), dtype=np.float32)
    mask = np.zeros((2,256,256), dtype=np.uint8)
    img = img_as_float(img)

       
    img[color_index, rr, cc] = 1
    mask[1, rr, cc] = 1

    if debug == True:
        plt.subplot(1,2,1)
        plt.imshow(img)

        plt.subplot(1,2,2)
        plt.imshow(mask)
        
    return img, mask

def generate_colored_circle_img(sigma_range, center_extent, radius_extent, color_index = 1 , debug = False):
    center_coords = np.random.randint(*center_extent,2)
    radius = np.random.randint(radius_extent)
    

    rr, cc = disk((center_coords[0], center_coords[1]), radius)
    

    img = np.zeros((3,256,256), dtype=np.float32)
    mask = np.zeros((2,256,256), dtype=np.uint8)

    img = img_as_float(img)
      
    img[color_index, rr, cc] = 1
    mask[0, rr, cc] = 1

    if debug == True:
        plt.subplot(1,2,1)
        plt.imshow(img)

        plt.subplot(1,2,2)
        plt.imshow(mask)
        
    return img, mask