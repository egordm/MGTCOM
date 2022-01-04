FROM gcc:11.1-bullseye AS build

WORKDIR /build

# Build pretrain binaries
COPY ./src/PreTrain ./src/PreTrain
RUN cd /build/src/PreTrain && make

FROM tensorflow/tensorflow:1.9.0-gpu-py3

# Copy python scripts
WORKDIR /app
COPY . .


#ENTRYPOINT ["python", "run.py"]
CMD ["python", "-c", "import tensorflow as tf;tf.test.gpu_device_name()"]