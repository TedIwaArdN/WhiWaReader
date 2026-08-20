# WhiWa Token Download
# by Dropkick
# 8/20/2026

import os
import requests
from io import BytesIO
from PIL import Image

# Here's PATH to your token image save folder
TOKEN_SAVE_PATH = "D:/OtherFiles/WuwaShareReader/token_img"

os.makedirs(TOKEN_SAVE_PATH, exist_ok=True)

# DON'T CHANGE
# Download image from encore.moe
url_front = "https://api-v2.encore.moe/resource/Data/Game/Aki/UI/UIResources/Common/Image/IconMowing/"
url_mid = "T_IconMowing_"
url_end = ".webp"
img_end = ".png"

for i in range(0, 100):
    try:
        url = url_front + url_mid + str(i).zfill(2) + url_end
        print(f"Now downloading #{str(i).zfill(2)} of 99, from {url}")
        response = requests.get(url)
        response.raise_for_status()

        if response.status_code == 200:
            output_path = os.path.join(TOKEN_SAVE_PATH, url_mid + str(i).zfill(2) + img_end)
            image = Image.open(BytesIO(response.content))
            image.save(output_path, "PNG")
            print(f"Success! Saved to: {output_path}")

    except Exception as e:
        print(f"Failed -- {e}")

print("===============================")
print("ALL AVAILABLE TOKENS DOWNLOADED")
print("===============================")
