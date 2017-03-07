# acerta-abide
Deep learning using the ABIDE data

## Environment Setup
In order to run the deep learning model, you need to install [docker](https://docs.docker.com/engine/getstarted/step_one/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

## Data preparation

The first step is to download the dataset:

```bash
nvidia-docker run -it --rm \
    -v $(realpath data):/opt/acerta-abide/data \
    acerta-abide \
    python download_abide.py
```

This command will download the preprocessed CC-200 dataset from Amazon S3.

And compile the dataset into CV folds and by experiment.

```bash
nvidia-docker run --rm \
    -v $(realpath data):/opt/acerta-abide/data \
    acerta-abide \
    python prepare_data.py \
        --whole \
        --male \
        --threshold \
        --folds 10 \
    2> /dev/null
```

## Model training

```bash
nvidia-docker run --rm \
    -v $(realpath data):/opt/acerta-abide/data \
    acerta-abide \
    python nn.py \
        --whole \
        --male \
        --threshold \
        --folds 10 \
    2> /dev/null
```

## Model evaluation

```bash
nvidia-docker run --rm \
    -v $(realpath data):/opt/acerta-abide/data \
    acerta-abide \
    python nn_evaluate.py \
        --whole \
        --male \
        --threshold \
        --folds 10 \
    2> /dev/null
```
