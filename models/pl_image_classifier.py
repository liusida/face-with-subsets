import torch
from torch import nn
import torch.nn.functional as F

import torchmetrics
import pytorch_lightning as pl

class LitImageClassifier(pl.LightningModule):

    def __init__(self, input_dim=196608, output_dim=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.save_hyperparameters()
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = self.forward(x)
        y_hat = F.softmax(x, dim=1)
        loss = F.cross_entropy(y_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_hat, y))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
