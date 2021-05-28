#%%
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import os
import pickle
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
logger = WandbLogger(project="face")

#%%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="pl_transfer_vggface2_detached", help="model")
args = parser.parse_args()
logger.log_hyperparams(args)

from models.pl_image_classifier import LitImageClassifier as pl_image_classifier
from models.pl_transfer_densenet import LitImageClassifier as pl_transfer_densenet
from models.pl_transfer_vggface2 import LitImageClassifier as pl_transfer_vggface2
from models.pl_transfer_vggface2 import LitImageClassifier as pl_transfer_vggface2_detached
models = {
    "pl_image_classifier": pl_image_classifier,
    "pl_transfer_densenet": pl_transfer_densenet,
    "pl_transfer_vggface2": pl_transfer_vggface2,
    "pl_transfer_vggface2_detached": pl_transfer_vggface2_detached,
}
current_model_cls = models[args.model]

#%%
df_labels = pd.read_csv("./data-labels/ffhq_aging_labels.csv")
# keep only confident ones (20778 rows)
df_labels = df_labels[df_labels['age_group_confidence']>0.9]
df_labels = df_labels[df_labels['gender_confidence']>0.9]
# get targets
labels_age_group = df_labels['age_group'].unique().tolist()
targets = {}
targets_rev = {}
for i,l in enumerate(labels_age_group):
    targets[l] = i
    targets_rev[i] = l
with open("targets.pkl", "wb") as f:
    pickle.dump(targets, f) # {'0-2':0, ...}
with open("targets_rev.pkl", "wb") as f:
    pickle.dump(targets_rev, f) # {0:'0-2', ...}

#%%
#%%
df_male = df_labels[df_labels['gender']=='male']
df_female = df_labels[df_labels['gender']=='female']

#%%
def make_weights_for_balanced_classes(subset):
    nclasses = len(targets)
    df = subset.dataset.df.iloc[subset.indices]
    count = [0] * nclasses
    for k,v in targets.items():
        count[v] = len(df[df['age_group']==k])
    print(count)
    len_df = len(df)
    print(len_df)
    weight = [0] * len_df
    i = 0
    for index, row in df.iterrows():
        weight[i] = len_df * 1. / count[targets[row['age_group']]]
        i+=1
    return weight, len_df

#%%
class MySubset:
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

set_male = MySubset(df_male)
l = len(set_male)
trainset_subset, valset_subset = torch.utils.data.random_split(set_male, [l-1000, 1000])

train_samples_weight, train_len_df = make_weights_for_balanced_classes(trainset_subset)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weight, train_len_df)
val_samples_weight, val_len_df = make_weights_for_balanced_classes(valset_subset)
val_sampler = torch.utils.data.sampler.WeightedRandomSampler(val_samples_weight, val_len_df)

for i, (data, target) in enumerate(trainset_subset):
    print(i,data.shape,target)
    if i>3: break
#%%
train_loader = torch.utils.data.DataLoader(
    trainset_subset,
    sampler=train_sampler,
    batch_size=256,
    num_workers=6,
    shuffle=False
)
val_loader = torch.utils.data.DataLoader(
    valset_subset,
    sampler=val_sampler,
    batch_size=256,
    num_workers=6,
    shuffle=False
)
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(target)
#     break
# %%
model = current_model_cls()

# checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='checkpoints/')

trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=40, val_check_interval=1.0, callbacks=[])
trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
# %%
trainer.save_checkpoint("pl-model.ckpt")
