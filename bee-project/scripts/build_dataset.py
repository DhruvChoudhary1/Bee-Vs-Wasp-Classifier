import os
import shutil
import random

DEST = "data"

for cls in ["bee", "wasp"]:
    os.makedirs(f"{DEST}/train/{cls}", exist_ok=True)
    os.makedirs(f"{DEST}/val/{cls}", exist_ok=True)

def get_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                paths.append(os.path.join(root, f))
    return paths

# Bee images
bee_images = get_images("data_raw/bee_vs_wasp/kaggle_bee_vs_wasp/bee1") + \
             get_images("data_raw/bee_vs_wasp/kaggle_bee_vs_wasp/bee2")

# Wasp images
wasp_images = get_images("data_raw/bee_vs_wasp/kaggle_bee_vs_wasp/wasp1") + \
              get_images("data_raw/bee_vs_wasp/kaggle_bee_vs_wasp/wasp2")

def split_and_copy(images, cls):
    random.shuffle(images)
    split = int(0.8 * len(images))

    for img in images[:split]:
        shutil.copy(img, f"{DEST}/train/{cls}")

    for img in images[split:]:
        shutil.copy(img, f"{DEST}/val/{cls}")

split_and_copy(bee_images, "bee")
split_and_copy(wasp_images, "wasp")

print("Bee vs Wasp dataset ready!")