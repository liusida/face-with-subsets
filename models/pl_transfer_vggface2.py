import torch
from torch import nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

import torchmetrics
import pytorch_lightning as pl

class LitImageClassifier(pl.LightningModule):

    def __init__(self, output_dim=10): # input_dim = [3, 256, 256]
        super().__init__()

        self.vggface2 = InceptionResnetV1(pretrained='vggface2').eval()
        for param in self.vggface2.parameters():
            param.requires_grad = False
            
        self.fc1 = nn.Linear(512, output_dim)

        self.save_hyperparameters()
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.vggface2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_hat, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
