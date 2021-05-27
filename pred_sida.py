import torchvision
import pytorch_lightning as pl
from models.pl_image_classifier import LitImageClassifier
import pickle
import pandas as pd

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
    pickle.dump(targets_rev, f) # {'0-2':0, ...}


with open("targets_rev.pkl", "rb") as f:
    targets_rev = pickle.load(f) # {'0-2':0, ...}

x = torchvision.datasets.folder.pil_loader("test-assets/sida.png")
x = torchvision.transforms.ToTensor()(x)
x = torchvision.transforms.Resize([256,256])(x)
x = x.unsqueeze(0)
print(x.shape)

model = LitImageClassifier.load_from_checkpoint("pl-model.ckpt")
# model = LitImageClassifier.load_from_checkpoint("checkpoints/tmp.ckpt")

y_hat = model(x)
y_hat = y_hat.flatten().detach().numpy().tolist()
for i in range(len(y_hat)):
    print(f"{targets_rev[i]} : {y_hat[i]:.0%}")
