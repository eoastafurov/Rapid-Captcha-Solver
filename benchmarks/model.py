from __future__ import annotations
from ctypes import Union
from distutils.errors import LibError
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
import torch.optim as optim 
import torch
import numpy as np
from typing import Optional, Literal, Dict, Any
from torchvision import models




class ModelLoader:
    def __init__(self) -> None:
        self.CATALOGUE = {
            'MobileNetV3-Small': (ModelLoader.load_mobilenet, 'small'),
            'MobileNetV3-Large': (ModelLoader.load_mobilenet, 'large'),
            'EfficientNet-B0': (ModelLoader.load_efficient, 'b0'),
            'ResNet18': (ModelLoader.load_resnet, 'resnet18')
        }

    def load(
        self,
        model_name: str,
        n_classes: int,
        weights: Literal['none', 'imagenet']
    ):
        if model_name not in self.CATALOGUE.keys():
            raise ValueError
        
        func, family = self.CATALOGUE[model_name]
        return func(family=family, n_classes=n_classes, weights=weights)

    @staticmethod
    def load_resnet(
        n_classes: int,
        weights: Literal['none', 'imagenet'],
        family: Literal['resnet18']
    ) -> torch.nn.Module:
        model = torch.hub.load(
            'pytorch/vision:v0.10.0', 
            family, 
            pretrained=True if weights == 'imagenet' else False
        )
        model.fc = torch.nn.Linear(in_features=512, out_features=n_classes)
        return model


    @staticmethod
    def load_efficient(
        n_classes: int, 
        weights: Literal['none', 'imagenet'],
        family: Literal['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
    ) -> torch.nn.Module:
        model = None
        if weights == 'imagenet':
            model = EfficientNet.from_pretrained('efficientnet-{}'.format(family), num_classes=n_classes)
        elif weights == 'none':
            original_dropout_rate = 0.5 * int(family[1]) / 7.0
            model = EfficientNet.from_name(
                'efficientnet-{}'.format(family),
                num_classes=n_classes,
                dropout_rate=original_dropout_rate,
                batch_norm_momentum=0.90
            )

        return model

    @staticmethod
    def load_mobilenet(
        n_classes: int,
        weights: Literal['none', 'imagenet'],
        family: Literal['small', 'large']
    ) -> torch.nn.Module:
        model = None
        if family == 'small':
            if weights == 'imagenet':
                model = models.mobilenet_v3_small(pretrained=True)
            elif weights == 'none':
                model = models.mobilenet_v3_small(pretrained=False)
            
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=576, out_features=400, bias=True),
                torch.nn.Hardswish(),
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=400, out_features=n_classes, bias=True)
            )
        elif family == 'large':
            if weights == 'imagenet':
                model = models.mobilenet_v3_large(pretrained=True)
            elif weights == 'none':
                model = models.mobilenet_v3_large(pretrained=False)
            
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=960, out_features=400, bias=True),
                torch.nn.Hardswish(),
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=400, out_features=n_classes, bias=True)
            )

        return model

class CaptchaLight(pl.LightningModule):
    def __init__(
        self, 
        model_name: Literal['MobileNetV3-Small', 'MobileNetV3-Large', 'EfficientNetB0'],
        weights: Literal['none', 'imagenet'],
        n_classes: int,
        class_weights: Union[torch.Tensor, None],
        class_map: Dict[str, str],
        learning_rate: float,
        scheduler_patience: int,
        data_hparams: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = ModelLoader().load(
            model_name=model_name,
            n_classes=n_classes,
            weights=weights
        )
        self.class_map = class_map
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.data_hparams = data_hparams
        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights
        )

    # @torch.jit.script
    def forward(self, inputs) -> Any:
        return self.backbone(inputs)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.25,
            patience=self.scheduler_patience,
            threshold=1e-2,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-09,
            verbose=True
        )
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        }
        return [optimizer], [lr_dict]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        accuracy = torch.sum(y_hat.argmax(dim=1) == y) / y.shape[0]

        logs = {'train_loss': loss.detach().cpu().numpy(), 'train_accuracy': accuracy.detach().cpu().numpy()}

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=False, prog_bar=True)

        return {'loss': loss, 'accuracy': accuracy.detach(), 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        accuracy = torch.sum(y_hat.argmax(dim=1) == y) / y.shape[0]

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return {'val_loss': loss, 'val_accuracy': accuracy.detach()}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        accuracy = torch.sum(y_hat.argmax(dim=1) == y) / y.shape[0]

        self.log("test_loss", loss, on_step=True, on_epoch=False)
        self.log('test_accuracy', accuracy, on_step=True, on_epoch=False)
        return {'test_loss': loss, 'test_accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_accuracy': avg_acc}

        print('\n\nVAL Accuracy: {}\nVAL Loss: {}\n'.format(
            round(float(avg_acc), 3),
            avg_loss
        ))

        return {'val_loss': avg_loss, 'val_accuracy': avg_acc, 'log': tensorboard_logs}

    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     avg_acc = torch.stack([x['test_accuracy'] for x in outputs]).mean()

    #     tensorboard_logs = {'test_loss': avg_loss, 'test_accuracy': avg_acc}
    #     self.log('test_accuracy', avg_acc, on_epoch=True, on_step=False)

    #     return {'test_loss': avg_loss, 'test_accuracy': avg_acc, 'log': tensorboard_logs}

    # def training_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()

    #     self.log('train_accuracy', avg_acc, on_epoch=True, on_step=False, logger=True)
    #     self.log('train_loss', avg_loss, on_epoch=True, on_step=False, logger=True)

    #     return None


