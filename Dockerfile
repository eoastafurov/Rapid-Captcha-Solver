FROM pytorch/pytorch:latest

ADD requirements.txt /workspace
ADD benchmarks/ /workspace

RUN pip install -r requirements.txt
RUN pip install protobuf==3.20
RUN pip install onnxruntime