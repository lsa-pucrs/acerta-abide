# acerta-abide
Deep learning using the ABIDE data

## Environment Setup
In order to run the deep learning model, you may want to install [docker](https://docs.docker.com/engine/getstarted/step_one/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to an easy setup.

If you want to install it directly in your machine, you need to install [CUDA 8.0](https://developer.nvidia.com/cuda-downloads) and the packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Running with separate virtual environments

You can also keep a separate set of python libraries to work with the code (if you are not root, or if you have multiple python set ups in the same machine) by running the following commands.
```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

Just remember to always activate your separate environment before running the code by running
```bash
source env/bin/activate
```

## Running with Docker
First, you need to build the project image: ```docker build -t acerta-abide .```
Second, you need to start a container with this image, in order to execute the next steps: ```nvidia-docker run -it --rm -v $(realpath .):/opt/acerta-abide acerta-abide /bin/bash```

## Data preparation

The first step is to download the dataset:

```bash
python download_abide.py
```

This command will download the preprocessed datasets from Amazon S3.

And compile the dataset into CV folds and by experiment.

```bash
python prepare_data.py \
    --whole \
    --male \
    --threshold \
    --folds 10 \
    cc200
```

## Model training

```bash
python nn.py \
    --whole \
    --male \
    --threshold \
    --folds 10 \
    cc200
```

## Model evaluation

```bash
python nn_evaluate.py \
    --whole \
    --male \
    --threshold \
    --folds 10 \
    --mean
    cc200 \
```
