from argparse import ArgumentParser

from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from video_dataset import VideoDataset
from pytorch_i3d import InceptionI3d

import pytorch_lightning as pl

import pdb

class i3DTrainer(pl.LightningModule):
    def __init__(self, hparams, load_pretrained_charades=False):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.i3d = InceptionI3d(400, in_channels=3)
        if load_pretrained_charades:
            if self.hparams.model_path != '':
                print('Loading %s weights ' % self.hparams.model_path)
                self.i3d.replace_logits(157)
                self.i3d.load_state_dict(torch.load(self.hparams.model_path))
        self.i3d.replace_logits(self.hparams.num_classes)
        # freeze all layers
        for params in self.i3d.parameters():
            params.requires_grad = False
        # unfreeze last logits layer
        for params in self.i3d.logits.parameters():
            params.requires_grad = True

    def forward(self, batch):
        inputs, labels, vidx = batch
        per_frame_logits = self.i3d(inputs)
        predictions = torch.max(per_frame_logits, dim=2)[0]
        predictions = torch.argmax(predictions, axis=1)
        return per_frame_logits, predictions

    def calculate_loss(self, batch, per_frame_logits):
        inputs, labels, vidx = batch
        per_frame_logits = torch.max(per_frame_logits, dim=2)[0]

        # compute classification loss (with max-pooling)
        loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)

        return loss

    def training_step(self, batch, batch_idx):
        per_frame_logits, predictions = self(batch)
        loss = self.calculate_loss(batch, per_frame_logits)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("train_loss", avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, weight_decay=0.0000001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
        return [optimizer], [scheduler]

    def train_dataloader(self):
        clip_transform = transforms.Compose([transforms.RandomCrop(448),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Resize(224)])
        train_dataset = VideoDataset(self.hparams.train_root, self.hparams.train_labels,
                                     transform=clip_transform, num_classes=self.hparams.num_classes)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=True, num_workers=self.hparams.n_threads, pin_memory=True)
