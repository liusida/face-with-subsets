import torchvision
import pytorch_lightning as pl
from models.pl_image_classifier import LitImageClassifier
import pickle, argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", default="test-assets/sida.png", help="file path")
args = parser.parse_args()

with open("targets_rev.pkl", "rb") as f:
    targets_rev = pickle.load(f) # {'0-2':0, ...}

x = torchvision.datasets.folder.pil_loader(args.filename)
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
