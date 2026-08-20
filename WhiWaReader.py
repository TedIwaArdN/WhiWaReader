# Wuwa Share Reader -- WhiWa Reader
# by Dropkick
# 8/20/2026

import cv2
import numpy as np
import os
# need to execute
# pip install scikit-image
from skimage.metrics import structural_similarity as ssim

# here goes PATH of character avatar
AVATAR_FOLDER_PATH = "d:/OtherFiles/WuwaShareReader/waves_avatar"

#
# Get character avatars
#
image_files = sorted([f for f in os.listdir(AVATAR_FOLDER_PATH) if f.lower().endswith((".png"))])
img_data = []

# Initialization, can run only once to save power
def init():
    for i, img_name in enumerate(image_files):
        img = cv2.imread(AVATAR_FOLDER_PATH + "/" + img_name, cv2.IMREAD_UNCHANGED)

        # RGBA to GRAYSCALE
        if img.shape[2] == 4:
            b, g, r, alpha = cv2.split(img)
            alpha_f = alpha.astype(float) / 255.0
            
            b_black = (b.astype(float) * alpha_f).astype(np.uint8)
            g_black = (g.astype(float) * alpha_f).astype(np.uint8)
            r_black = (r.astype(float) * alpha_f).astype(np.uint8)
            
            img = cv2.merge((b_black, g_black, r_black))

        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_data.append(img_gray)

#
# Function to get info
# Input: PATH to image
# Output: list of 6, name of images of WhiWa Resonators
#
def ReadWhiWaShare(WhiWaImg_PATH):
    res = []
    # Read image
    img_WhiWa = cv2.imread(WhiWaImg_PATH)
    # Resonator image top-left corner & dimension
    resonators = [(1260, 416), (1369, 416), (1478, 416), 
                  (1260, 572), (1369, 572), (1478, 572)]
    w, h = 91, 91

    for i, (x, y) in enumerate(resonators):
        # Extract current character avatar & to grayscale
        sub_image = img_WhiWa[y:y+h, x:x+h]
        sub_gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)

        # Fine best match
        max_score = -1
        max_index = 0

        # Compare current character with all avatars
        for index, gray in enumerate(img_data):
            gray = cv2.resize(gray, (w, h))
            score, diff = ssim(
                sub_gray,
                gray,
                gaussian_weights=True,
                sigma=1.5,
                full=True,
                data_range=sub_gray.max() - sub_gray.min()
            )
            # Get the best one
            if score > max_score:
                max_score = score
                max_index = index

        res.append(image_files[max_index])

    return res

#
# sample run:
#
init()
print(ReadWhiWaShare("d:/OtherFiles/WuwaShareReader/WhiWa.png"))
