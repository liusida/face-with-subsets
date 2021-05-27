#%%
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import os
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
logger = WandbLogger(project="face")
from models.pl_image_classifier import LitImageClassifier

#%%
df_labels = pd.read_csv("./data-labels/ffhq_aging_labels.csv")
# keep only confident ones (20778 rows)
df_labels = df_labels[df_labels['age_group_confidence']>0.9]
df_labels = df_labels[df_labels['gender_confidence']>0.9]
# get targets
labels_age_group = df_labels['age_group'].unique().tolist()
targets = {}
for i,l in enumerate(labels_age_group):
    targets[l] = i
targets # {'0-2':0, ...}

#%%
#%%
df_male = df_labels[df_labels['gender']=='male']
df_female = df_labels[df_labels['gender']=='female']

#%%
class Subset:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_number = self.df["image_number"].iloc[index]
        filename = f"./data/small/{image_number:05}.png"
        x = torchvision.datasets.folder.pil_loader(filename)
        x = torchvision.transforms.ToTensor()(x)
        x = torchvision.transforms.Resize([256,256])(x)
        y = self.df["age_group"].iloc[index]
        y = targets[y]
        y = torch.tensor(y)
        return x,y
    def __len__(self) -> int:
        return len(self.df)

set_male = Subset(df_male)
l = len(set_male)
trainset_male, valset_male = torch.utils.data.random_split(set_male, [l-1000, 1000])

for i, (data, target) in enumerate(trainset_male):
    print(i,data.shape,target)
    if i>10: break
#%%
train_loader = torch.utils.data.DataLoader(
    trainset_male,
    batch_size=128,
    num_workers=6,
    shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    valset_male,
    batch_size=128,
    num_workers=6,
    shuffle=True
)
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(target)
#     break
# %%
model = LitImageClassifier()
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=20, val_check_interval=5)
trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
# %%
