import os
import pandas as pd
from medmnist import PneumoniaMNIST
from os.path import join as pjoin

root = "tests/test_dataset/pneumonia_data"
os.makedirs(root, exist_ok=True)
os.makedirs(f"{root}/images", exist_ok=True)

data = PneumoniaMNIST(split="train", download=True, root=root)

records = []
for idx, (img, label) in enumerate(data):
    path = f"images/{idx}.png"
    img.save(pjoin(root, path))
    records.append([idx, pjoin(os.getcwd(), root, path), int(label[0])])

df = pd.DataFrame(records, columns=["sample_id", "image_path", "outcome"])
df.to_csv(f"{root}/metadata.csv", index=False)