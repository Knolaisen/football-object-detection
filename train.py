# %%
import os
from ultralytics import YOLO
import wandb
from dotenv import load_dotenv

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))


# %%
model: YOLO = YOLO("yolo11l.pt")
data_path = "data/data.yaml"

# %%
model.val(
    data=data_path
)

# %%
results = model.train(
    data=data_path,
    epochs=300,
    batch=16,
    device="0",
    workers=8,
    project="CV-RBK-large",
    name="baseline",
    exist_ok=True,
    save_period=1,
    fliplr=0.5,
)
#  /home/krisnol/Documents/data/rbk_structured/train/images
#  /home/krisnol/Documents/football-object-detection/data/rbk_structured/val/images
# 
model.val(
    data=data_path,
)

# %%
results = model.predict("test_images/21.jpg", save=True)

# %%
results[0].show()



