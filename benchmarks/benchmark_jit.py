import torch
import pytorch_lightning as pl
from transforms import IBotWeakTransformCollection
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from  torch.cuda.amp import autocast
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import gc
import time
import json

from model import CaptchaLight
from data import DatasetImageFolder, DataHyperParams
from transforms import IBotWeakTransformCollection


WEIGHTS = {
    'MobileNetV3-Small': 'jit_weights/mbnsmall.pt',
    'MobileNetV3-Large': 'jit_weights/mbnlarge.pt',
    'ResNet18': 'jit_weights/resnet18.pt'
}
INFERENCE_DATA_PATH = 'inference_data/'
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
IMG_SIZE = 128


def configure_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Possibles: MobileNetV3-Small, MobileNetV3-Large, ResNet18')
    parser.add_argument('--num_workers', type=str, required=True, help='Dataloader num workers')
    return parser


def load_model(path, model_name):
    scripted_module = torch.jit.load(path)
    return scripted_module


def load_dataset():
    data_hparams=DataHyperParams(
        class_names=CLASS_NAMES,
        img_size=IMG_SIZE,
        num_classes=None,
        batch_size=None,
        train_data_path=None,
        val_data_path=None,
        test_data_path=None
    )
    dataset = DatasetImageFolder(
        root=INFERENCE_DATA_PATH,
        transform=IBotWeakTransformCollection.validation_transform(size_=IMG_SIZE),
        data_hyper_params=data_hparams
    )
    return dataset


def benchmark(model, dataset, model_name, min_bs = 1, max_bs = 27, num_workers = 2, max_iters = 50):
    raw_results = {}
    pbar = tqdm(total=(max_bs - min_bs) // 9 + 1)
    for bs in range(min_bs, max_bs + 1):
        if bs % 9 != 0:
            continue
        gc.collect()
        pbar.set_description('JIT {}: Batch size = {}'.format(model_name, bs))
        dataloader = DataLoader(dataset, num_workers=num_workers, shuffle=False, batch_size=bs)
        raw_results[bs] = []
        for idx, batch in enumerate(dataloader):
            if idx > max_iters:
                break
            gc.collect()
            inputs, targets = batch
            time_begin = time.time()
            outputs = model(inputs)
            elapsed = time.time() - time_begin
            throughput = bs / elapsed
            raw_results[bs].append(throughput)
            gc.collect()
        pbar.update(1)
    pbar.close()
    return raw_results


if __name__ == '__main__':
    parser = configure_parser()
    args = parser.parse_args()

    model_name = args.model_name
    model_path = WEIGHTS[model_name]
    model = load_model(model_path, model_name)
    dataset = load_dataset()

    results = benchmark(
        model=model, 
        dataset=dataset,
        model_name=model_name,
        num_workers=int(args.num_workers)
    )
    with open('results/jit/{}.json'.format(''.join(args.model_name.split('-'))), 'w') as f:
        json.dump(results, f)

