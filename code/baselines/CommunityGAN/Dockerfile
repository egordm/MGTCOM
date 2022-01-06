FROM gcc:11.1-bullseye AS build

WORKDIR /build

# Build pretrain binaries
COPY ./src/PreTrain ./src/PreTrain
RUN cd /build/src/PreTrain && make

FROM tensorflow/tensorflow:1.9.0-gpu-py3

RUN apt update && apt -y install libomp-dev

WORKDIR /app

# Copy the pretrain binaries
COPY --from=build /build/src/PreTrain/magic /app/src/PreTrain/magic

# Copy python scripts
COPY . .

ENTRYPOINT ["python", "run.py"]
