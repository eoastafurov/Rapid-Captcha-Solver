python benchmark_torch.py --model_name ResNet18 --num_workers 0
python benchmark_torch.py --model_name MobileNetV3-Small --num_workers 0
python benchmark_torch.py --model_name MobileNetV3-Large --num_workers 0


python benchmark_onnx.py --model_name MobileNetV3-Small --num_workers 0
python benchmark_onnx.py --model_name MobileNetV3-Large --num_workers 0
python benchmark_onnx.py --model_name ResNet18 --num_workers 0

python benchmark_jit.py --model_name MobileNetV3-Small --num_workers 0
python benchmark_jit.py --model_name MobileNetV3-Large --num_workers 0
python benchmark_jit.py --model_name ResNet18 --num_workers 0
