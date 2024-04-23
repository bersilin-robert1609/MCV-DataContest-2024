# %%
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# %%
IMAGES_DIR = "../data/images"
DIRS = ["train", "test", "val"]

# %%
train_imgs = os.listdir(os.path.join(IMAGES_DIR, "train"))

# %%
# Gamma correction on an image

def adjust_gamma(image, gamma=1.0):
    """
    Apply gamma correction to an image.

    Parameters:
        image: numpy array, input image of shape (W, H, 3).
        gamma: float, gamma value. Default is 1.0.

    Returns:
        numpy array, gamma corrected image.
    """
    image = image.astype(np.float32) / 255.0
    gamma_corrected = np.power(image, gamma)
    gamma_corrected = np.clip(gamma_corrected, 0, 1)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
    return gamma_corrected

# %%
def plot_images(title, imgs):
    subplots = len(imgs)
    plt.figure(figsize=(15, 5))
    plt.title(title)
    plt.axis("off")
    
    for i, img in enumerate(imgs):
        plt.subplot(1, subplots, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()

# %%
GAMMA = 0.4

OUTPUT_DIR = "../data_3/images"
for d in DIRS:
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)
    
for d in DIRS:
    FOLDER = os.path.join(IMAGES_DIR, d)
    
    imgs = os.listdir(FOLDER)
    
    for i, img in enumerate(imgs):
        if(os.path.exists(os.path.join(OUTPUT_DIR, d, img))):
            print(f"Image: {img} already done")
            continue
        
        image = cv.imread(os.path.join(FOLDER, img), cv.IMREAD_COLOR)
        
        gamma_corrected = adjust_gamma(image, GAMMA)
        
        cv.imwrite(os.path.join(OUTPUT_DIR, d, img), gamma_corrected)
        
        print(f"{i}, Image: {img} done")        

# %%



