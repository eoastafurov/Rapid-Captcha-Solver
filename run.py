import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor, GradientAccumulationScheduler
import torch
import json

from dataset import IBotDataModule, DataHyperParams
from captcha_model import CaptchaLight
from transforms import IBotWeakTransformCollection

import random
random.seed(42)
import numpy as np
np.random.seed(42)


MODEL_NAME = 'MobileNetV3-Large'
WEIGHTS = 'imagenet'
RUN_NAME = 'MBN-Large-weighted'
SCHEDULER_PATIENCE = 5

BATCH_SIZE = 250
IMG_SIZE = 128
TRAIN_DATA_PATH = '/home/eugeny/ibot-ml/Small/train'
VAL_DATA_PATH = '/home/eugeny/ibot-ml/Small/val'
TEST_DATA_PATH = '/home/eugeny/ibot-ml/Small/val'
CLASS_NAMES = [
    'airplane',
    'bicycle',
    'boat',
    'bus',
    'car',
    'floatplane',
    'train',
    'truck',
    'umbrella',
    'van'
]



if __name__ =='__main__':
    # init dm
    dm = IBotDataModule(
        data_hparams=DataHyperParams(
            num_classes=len(CLASS_NAMES),
            class_names=CLASS_NAMES,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            train_data_path=TRAIN_DATA_PATH,
            val_data_path=VAL_DATA_PATH,
            test_data_path=TEST_DATA_PATH
        ),
        transform_collection=IBotWeakTransformCollection
    )
    dm.setup()
    class_weights = torch.Tensor(dm.train_data.class_weights)
    class_weights /= torch.max(class_weights)
    class_weights = 1 / class_weights
    # print('Class Weights: {}'.format(class_weights))

    # Init Logger
    logger = TensorBoardLogger('lightning_logs', name=RUN_NAME)

    model = CaptchaLight(
        model_name=MODEL_NAME,
        weights=WEIGHTS,
        n_classes=len(CLASS_NAMES),
        class_map={key: idx for idx, key in enumerate(CLASS_NAMES)},
        learning_rate=3e-4,
        scheduler_patience=SCHEDULER_PATIENCE,
        data_hparams=None,
        class_weights=class_weights
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor='val_loss',
        mode='min',
        dirpath='runs/{}/'.format(RUN_NAME),
        filename='ibot-{epoch:02d}-{step:d}-{val_loss:.4f}',
        save_last=True,
        verbose=True,
        every_n_epochs=1
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=20,
        verbose=True,
        strict=True,
    )

    weight_averaging_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    accumulator = GradientAccumulationScheduler(scheduling={0: 4, 15: 2, 25: 1})

    # Initialize a Trainer
    trainer = pl.Trainer(
        gpus=1,
        # strategy='ddp',
        precision=16,
        max_epochs=150,
        min_epochs=1,
        progress_bar_refresh_rate=1,
        callbacks=[checkpoint_callback, early_stopping_callback, weight_averaging_callback, lr_monitor, accumulator],
        check_val_every_n_epoch=1,
        logger=logger,
        log_every_n_steps=1,
        flush_logs_every_n_steps=50,
    )

    # Train the model ⚡
    torch.cuda.empty_cache()
    trainer.fit(model, dm)

    # result = trainer.test(model, dm)

