import os
import random
import shutil

IMAGE_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2

images = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))]
random.shuffle(images)

train_cutoff = int(len(images) * TRAIN_RATIO)
val_cutoff = int(len(images) * (TRAIN_RATIO + VAL_RATIO))

splits = {
    "train": images[:train_cutoff],
    "val": images[train_cutoff:val_cutoff],
    "test": images[val_cutoff:]
}

for split, files in splits.items():
    os.makedirs(f"{IMAGE_DIR}/{split}", exist_ok=True)
    os.makedirs(f"{LABEL_DIR}/{split}", exist_ok=True)

    for file in files:
        shutil.move(f"{IMAGE_DIR}/{file}", f"{IMAGE_DIR}/{split}/{file}")
        label = file.rsplit(".", 1)[0] + ".txt"
        shutil.move(f"{LABEL_DIR}/{label}", f"{LABEL_DIR}/{split}/{label}")
