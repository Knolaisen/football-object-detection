import os
import shutil
import pandas as pd

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


path_1 = "data/rbk/1_train-val_1min_aalesund_from_start/gt/gt.txt"
path_2 = "data/rbk/2_train-val_1min_after_goal/gt/gt.txt"
labels_folder_path = "rbk_structured/train/labels/"

# Read the files
dataframe_1 = pd.read_csv(path_1, sep=",", header=None)
dataframe_2 = pd.read_csv(path_2, sep=",", header=None)

# Correct the indices
dataframe_2[0] = dataframe_2[0] + 1802
combined = pd.concat([dataframe_1, dataframe_2])

# Rename the columns and drop unnecessary ones
combined.rename(columns={0: "frame", 1: "object_number", 2: "x_left", 3: "y_top", 4: "width", 5: "height", 6: "off_screen", 7: "class", 8: "idk"}, inplace=True)
combined.drop(columns=["off_screen", "idk"], inplace=True)

# Go from MOT to yolo ones
combined["x_center"] = combined["x_left"] + combined["width"]/2
combined["y_center"] = combined["y_top"] + combined["height"]/2 

# normalize the coordinates
combined["x_center"] = combined["x_center"] / 1920
combined["y_center"] = combined["y_center"] / 1080
combined["width"] = combined["width"] / 1920
combined["height"] = combined["height"] / 1080

def save_to_file(dataframe, picture_id):
    dataframe = dataframe[["class", "x_center", "y_center", "width", "height"]]
    dataframe["class"] = dataframe["class"] - 1
    dataframe.to_csv(f"data/rbk_structured/labels/train/{picture_id}.txt", header=False, index=False, sep=" ")

for picture_id, dataframe in combined.groupby("frame"):
    save_to_file(dataframe, picture_id)