FROM gcr.io/tensorflow/tensorflow:latest-gpu

COPY . /opt/acerta-abide
WORKDIR /opt/acerta-abide

RUN pip install -r requirements.txt