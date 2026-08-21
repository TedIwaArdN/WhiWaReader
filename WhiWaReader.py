# Wuwa Share Reader -- WhiWa Reader
# by Dropkick
# 8/21/2026

import numpy as np
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# here goes PATH of character avatar & token image
AVATAR_FOLDER_PATH = "d:/OtherFiles/WuwaShareReader/waves_avatar"
TOKEN_FOLDER_PATH = "d:/OtherFiles/WuwaShareReader/token_img"

#
# Get resonator avatars & token
#
image_files = sorted([f for f in os.listdir(AVATAR_FOLDER_PATH) if f.lower().endswith((".png"))])
img_data = []

token_files = sorted([f for f in os.listdir(TOKEN_FOLDER_PATH) if f.lower().endswith((".png"))])
token_img = []

# Initialization, can run only once to save power
def init():
    # Resonator
    for i, img_name in enumerate(image_files):
        img = Image.open(AVATAR_FOLDER_PATH + "/" + img_name)
        img_gray = img.convert("LA")
        gray, alpha = img_gray.split()
        bg = Image.new("L", img_gray.size, 0)
        bg.paste(gray, mask=alpha)
        img_gray=bg
        img_data.append(img_gray)

    # Token
    for i, token_name in enumerate(token_files):
        img = Image.open(TOKEN_FOLDER_PATH + "/" + token_name)
        img_gray = img.convert("LA")
        gray, alpha = img_gray.split()
        bg = Image.new("L", img_gray.size, 0)
        bg.paste(gray, mask=alpha)
        img_gray=bg
        token_img.append(img_gray)

#
# Function to get info
# Input: PATH to image
# Output: list of 6, name of images of WhiWa Resonators
#
def ReadWhiWaShare(WhiWaImg_PATH):
    res_resonator = []
    res_token = []

    # Read image
    img_WhiWa = np.array(Image.open(WhiWaImg_PATH))
    # Resonator image top-left corner & dimension
    resonators = [(1260, 416), (1369, 416), (1478, 416), 
                  (1260, 572), (1369, 572), (1478, 572)]
    # Token image top-left corner & dimension
    tokens = [(1587, 416), (1587, 572)]
    w, h = 91, 91

    # Resonator detection
    for i, (x, y) in enumerate(resonators):
        # Extract current character avatar & to grayscale
        sub_image = Image.fromarray(img_WhiWa[y:y+h, x:x+h])
        sub_gray = sub_image.convert("L")
        sub_gray = np.array(sub_gray)

        # Fine best match
        max_score = -1
        max_index = 0

        # Compare current character with all avatars
        for index, gray in enumerate(img_data):
            gray.thumbnail((w, h), resample=Image.Resampling.LANCZOS)
            gray = np.array(gray)
            
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

        res_resonator.append(image_files[max_index])

    # Token detection
    ###
    for i, (x, y) in enumerate(tokens):
        # Extract current character avatar & to grayscale
        sub_image = Image.fromarray(img_WhiWa[y:y+h, x:x+h])
        sub_gray = sub_image.convert("L")
        sub_gray = np.array(sub_gray)

        # Fine best match
        max_score = -1
        max_index = 0

        # Compare current token with all tokens
        for index, gray in enumerate(token_img):
            gray.thumbnail((w, h), resample=Image.Resampling.LANCZOS)
            gray = np.array(gray)

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

        res_token.append(token_files[max_index])

    return res_resonator, res_token

#
# sample run:
#
init()
print(ReadWhiWaShare("d:/OtherFiles/WuwaShareReader/WhiWa.png"))
