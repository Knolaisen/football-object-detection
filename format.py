import os
import shutil

# Define paths
folder1 = "data/rbk/1_train-val_1min_aalesund_from_start/img1"
folder2 = "data/rbk/2_train-val_1min_after_goal/img1"
destination = "data/rbk_structured/images/train"

# Ensure destination folder exists
os.makedirs(destination, exist_ok=True)
index = 1
# Copy and rename files from folder1
for filename in os.listdir(folder1):
    src = os.path.join(folder1, filename)
    dst = os.path.join(destination, f"{index}.jpg")
    index += 1
    shutil.copy(src, dst)

# Copy and rename files from folder2
for filename in os.listdir(folder2):
    src = os.path.join(folder2, filename)
    dst = os.path.join(destination, f"{index}.jpg")
    index += 1
    shutil.copy(src, dst)

print("Merging complete!")