# %%
import os
import yaml
import pandas as pd

# %%
image_directory = "data/rbk_structured/images/train/"
labels_directory = "data/rbk_structured/labels/train/"
k_fold_directory = "k_fold/"



# %%
number_of_folds = 5
number_of_images = 3604

folder_prefix =  "/work/krisnol/football-object-detection/data/k_fold"

# %%
for i in range(number_of_folds):
    # create directories
    os.makedirs(os.path.join(folder_prefix, str(i)), exist_ok=True)
    os.makedirs(os.path.join(folder_prefix, str(i), "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(folder_prefix, str(i), "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(folder_prefix, str(i), "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(folder_prefix, str(i), "labels", "val"), exist_ok=True)

    # create data.yaml file
    data_yaml = {
        "train": "/work/krisnol/football-object-detection/data/k_fold/" + str(i) + "/images/train",
        "val": "/work/krisnol/football-object-detection/data/k_fold/" + str(i) + "/images/val",
        "nc": 2 ,
        "names": {0: "ball", 1: "player"}
    }

    data_yaml_path = os.path.join(folder_prefix, str(i), "data.yaml")
    with open(data_yaml_path, 'w') as file:
        yaml.dump(data_yaml, file, default_flow_style=False)


    train_indices = []
    val_indices = []
    for j in range(1, number_of_images +1):
        if j % number_of_folds == i:
            val_indices.append(j)
        else:
            train_indices.append(j)
    
    for index in train_indices:
        # copy images
        image_file = os.path.join(image_directory, str(index) + ".jpg")
        os.system(f"cp {image_file} {os.path.join(folder_prefix, str(i), 'images', 'train')}")
        
        # copy labels
        label_file = os.path.join(labels_directory, str(index) + ".txt")
        os.system(f"cp {label_file} {os.path.join(folder_prefix, str(i), 'labels', 'train')}")
    for index in val_indices:
        # copy images
        image_file = os.path.join(image_directory, str(index) + ".jpg")
        os.system(f"cp {image_file} {os.path.join(folder_prefix, str(i), 'images', 'val')}")
        
        # copy labels
        label_file = os.path.join(labels_directory, str(index) + ".txt")
        os.system(f"cp {label_file} {os.path.join(folder_prefix, str(i), 'labels', 'val')}")



