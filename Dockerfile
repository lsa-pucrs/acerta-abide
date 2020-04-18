FROM gcr.io/tensorflow/tensorflow:latest-gpu

WORKDIR /opt/acerta-abide
COPY requirements.txt /opt/acerta-abide
RUN pip install -r requirements.txt

COPY . /opt/acerta-abide
