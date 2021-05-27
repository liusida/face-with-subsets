from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import pandas as pd
import torch
import torchvision
import pytorch_lightning as pl
from models.pl_image_classifier import LitImageClassifier
from pytorch_lightning.loggers import WandbLogger
logger = WandbLogger(project="face")

# read labels
df_labels = pd.read_csv("./data-labels/ffhq_aging_labels.csv")
df_labels = df_labels[df_labels['age_group_confidence']>0.9]
labels_age_group = df_labels['age_group'].unique().tolist()
targets = {}
for i,l in enumerate(labels_age_group):
    targets[l] = i
targets

# divide into two subsets by gender
df_male = df_labels[df_labels['gender']=='male']
df_female = df_labels[df_labels['gender']=='female']

# dataset
class Subset:
    """
        prepare the dataset for DataLoader
        hard code for now
    """
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

# dataloader
train_loader = torch.utils.data.DataLoader(
    set_male,
    batch_size=64,
    num_workers=6,
    shuffle=True
)

# train
model = LitImageClassifier()
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=3)
trainer.fit(model, train_loader)