FROM maven:3.8.1-openjdk-15-slim

WORKDIR /app

RUN apt update && apt install -y python3 python3-pip
RUN pip3 install natsort

COPY . .

RUN mvn package

COPY run.py /app/run.py

ENTRYPOINT ["python3", "/app/run.py"]