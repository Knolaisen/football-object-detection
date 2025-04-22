# %%
import os
import wandb
from dotenv import load_dotenv

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

from ultralytics import YOLO


# %%
model: YOLO = YOLO("yolo11l.pt")
# data_path = "data.yaml"
data_path = "data/k_fold/0/data.yaml"

# %%
model.val(
    data=data_path
)

# %%
results = model.train(
    data=data_path,
    epochs=2000,
    batch=16,
    device="0",
    workers=8,
    project="CV-RBK-large",
    exist_ok=True,
    save_period=1,
    fliplr=0.5,
)
#  /home/krisnol/Documents/data/rbk_structured/train/images
#  /home/krisnol/Documents/football-object-detection/data/rbk_structured/val/images


