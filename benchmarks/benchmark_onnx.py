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
import onnxruntime
import os


from model import CaptchaLight
from data import DatasetImageFolder, DataHyperParams
from transforms import IBotWeakTransformCollection


WEIGHTS = {
    'MobileNetV3-Small': 'torch_weights/mbnsmall.ckpt',
    'MobileNetV3-Large': 'torch_weights/mbnlarge.ckpt',
    'ResNet18': 'torch_weights/resnet18.ckpt'
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
    model = CaptchaLight.load_from_checkpoint(path)
    assert model_name == model._hparams.model_name
    return model


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


def dump_and_load(model, sample_inputs):
    filepath = "tmp/tmp.onnx"
    model.to_onnx(filepath, sample_inputs, export_params=True)
    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name
    return ort_session, input_name, filepath


def benchmark(model, dataset, model_name, min_bs = 1, max_bs = 27, num_workers = 2, max_iters = 50):
    raw_results = {}
    pbar = tqdm(total=(max_bs - min_bs) // 9 + 1)
    for bs in range(min_bs, max_bs + 1):
        if bs % 9 != 0:
            continue
        gc.collect()
        pbar.set_description('ONNX {}: Batch size = {}'.format(model_name, bs))
        dataloader = DataLoader(dataset, num_workers=num_workers, shuffle=False, batch_size=bs)
        sample_inputs, _ = next(iter(dataloader))
        ort_session, input_name, filepath = dump_and_load(model, sample_inputs)
        raw_results[bs] = []
        for idx, batch in enumerate(dataloader):
            if idx > max_iters:
                break
            gc.collect()
            inputs, targets = batch
            ort_inputs = {input_name: inputs.numpy()}
            time_begin = time.time()

            ort_outs = ort_session.run(None, ort_inputs)

            elapsed = time.time() - time_begin
            throughput = bs / elapsed
            raw_results[bs].append(throughput)
            gc.collect()
        os.system('rm -f {}'.format(filepath))
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
    with open('results/onnx/{}.json'.format(''.join(args.model_name.split('-'))), 'w') as f:
        json.dump(results, f)

