# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 01:00:10 2020

@author: hp
"""

import cv2
import numpy as np
from guidedfilter import guided_filter

def get_illumination_channel(I, w):
    M, N, _ = I.shape
    padded = np.pad(I, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))

    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])
        brightch[i, j] = np.max(padded[i:i + w, j:j + w, :])

    return darkch, brightch

def get_atmosphere(I, brightch, p=0.1):
    M, N = brightch.shape
    flatI = I.reshape(M*N, 3)
    flatbright = brightch.ravel()

    searchidx = (-flatbright).argsort()[:int(M*N*p)]
    A = np.mean(flatI.take(searchidx, axis=0), dtype=np.float64, axis=0)
    return A

def get_initial_transmission(A, brightch):
    A_c = np.max(A)
    init_t = (brightch-A_c)/(1.-A_c)
    return (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t))

def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha, omega, w):
    im3 = np.empty(I.shape, I.dtype)
    for ind in range(0, 3):
        im3[:, :, ind] = I[:, :, ind] / A[ind]
    dark_c, _ = get_illumination_channel(im3, w)
    dark_t = 1 - omega*dark_c
    corrected_t = init_t
    diffch = brightch - darkch

    for i in range(diffch.shape[0]):
        for j in range(diffch.shape[1]):
            if(diffch[i, j] < alpha):
                corrected_t[i, j] = dark_t[i, j] * init_t[i, j]

    return np.abs(corrected_t)

def get_final_image(I, A, refined_t, tmin):
    refined_t_broadcasted = np.broadcast_to(refined_t[:, :, None], (refined_t.shape[0], refined_t.shape[1], 3))
    J = (I-A) / (np.where(refined_t_broadcasted < tmin, tmin, refined_t_broadcasted)) + A

    return (J - np.min(J))/(np.max(J) - np.min(J))

def dehaze(I, tmin, w, alpha, omega, p, eps, reduce=False):
    m, n, _ = I.shape
    Idark, Ibright = get_illumination_channel(I, w)
    A = get_atmosphere(I, Ibright, p)

    init_t = get_initial_transmission(A, Ibright) 
    if reduce:
        init_t = reduce_init_t(init_t)
    corrected_t = get_corrected_transmission(I, A, Idark, Ibright, init_t, alpha, omega, w)

    normI = (I - I.min()) / (I.max() - I.min())
    refined_t = guided_filter(normI, corrected_t, w, eps)
    J_refined = get_final_image(I, A, refined_t, tmin)
    
    enhanced = (J_refined*255).astype(np.uint8)
    f_enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
    f_enhanced = cv2.edgePreservingFilter(f_enhanced, flags=1, sigma_s=64, sigma_r=0.2)
    return f_enhanced

def reduce_init_t(init_t):
    init_t = (init_t*255).astype(np.uint8)
    xp = [0, 32, 255]
    fp = [0, 32, 48]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    init_t = cv2.LUT(init_t, table)
    init_t = init_t.astype(np.float64)/255
    return init_t


tmin = 0.1   # minimum value for t to make J image
w = 15       # window size, which determine the corseness of prior images
alpha = 0.4  # threshold for transmission correction
omega = 0.75 # this is for dark channel prior
p = 0.1      # percentage to consider for atmosphere
eps = 1e-3   # for J image

# im = cv2.imread('dark.png')
# orig = im.copy()

# I = np.asarray(im, dtype=np.float64) # Convert the input to an array.
# I = I[:, :, :3] / 255

# f_enhanced = dehaze(I, tmin, w, alpha, omega, p, eps)
# f_enhanced2 = dehaze(I, tmin, w, alpha, omega, p, eps, True)

import os

# IMG_FOLDERS = ["../data/images/train"]  
IMG_FOLDERS = ["../data/images/test", "../data/images/val"] 
# DEST_FOLDERS = ["../data_1/images/train_dehazed", "../data_1/images/test_dehazed", "../data_1/images/val_dehazed"]
# DEST_2_FOLDERS = ["../data_2/images/train_dehazed2"] 
DEST_2_FOLDERS = [ "../data_2/images/test_dehazed2", "../data_2/images/val_dehazed2"] 

# make destination folders

import time

for i in range(len(DEST_2_FOLDERS)):
    # if not os.path.exists(DEST_FOLDERS[i]):
    #     os.makedirs(DEST_FOLDERS[i])
    if not os.path.exists(DEST_2_FOLDERS[i]):
        os.makedirs(DEST_2_FOLDERS[i])
        
waste_set = ["3ad04319-5354-4f3a-b500-8326c817c77b.jpeg", "f6a4a427-9693-4c34-93a8-bc80ba2b2151.jpeg", "82e3d549-4057-4c6a-81b5-2658d5c25813.jpeg", "88322450-73c2-42e1-9ace-ee735207f02a.jpeg", "48a649f4-ca5c-499d-9183-f2b25b60deee.jpeg", "b4f94edf-f02e-420a-b435-638f6efcf6e9.jpeg"]

for i in range(len(IMG_FOLDERS)):
    imgs = os.listdir(IMG_FOLDERS[i])
    
    for img in imgs:
        
        if(os.path.exists(os.path.join(DEST_2_FOLDERS[i], img))):
            print("Image: ", img, "already exists in folder: ", DEST_2_FOLDERS[i])
            continue
        
        if img in waste_set:
            continue            
            
        start_time = time.time()
        print("Working on image: ", img, "from folder: ", IMG_FOLDERS[i])
        
        im = cv2.imread(os.path.join(IMG_FOLDERS[i], img))
        I = np.asarray(im, dtype=np.float64)
        
        I = I[:, :, :3] / 255
        
        # f_enhanced = dehaze(I, tmin, w, alpha, omega, p, eps)
        
        # time_2 = time.time()
        
        # print("Time taken for 1 image: ", img, "is: ", time_2 - start_time)
        
        f_enhanced2 = dehaze(I, tmin, w, alpha, omega, p, eps, True)
        
        # print("Saving dehazed image: ", img, "to folder: ", DEST_FOLDERS[i])
        
        # cv2.imwrite(os.path.join(DEST_FOLDERS[i], img), f_enhanced)
        cv2.imwrite(os.path.join(DEST_2_FOLDERS[i], img), f_enhanced2)
        
        end_time = time.time()
        
        print("Time taken for image: ", img, "is: ", end_time - start_time)        
    