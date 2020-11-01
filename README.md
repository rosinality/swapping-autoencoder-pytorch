# swapping-autoencoder-pytorch
Unofficial implementation of Swapping Autoencoder for Deep Image Manipulation (https://arxiv.org/abs/2007.00653) in PyTorch

## Usage

First create lmdb datasets:

> python prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... DATASET_PATH

This will convert images to jpeg and pre-resizes it. This implementation does not use progressive growing, but you can create multiple resolution datasets using size arguments with comma separated lists, for the cases that you want to try another resolutions later.

Then you can train model in distributed settings

> python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --batch BATCH_SIZE LMDB_PATH

train.py supports Weights & Biases logging. If you want to use it, add --wandb arguments to the script.

### Generate samples

You can test trained model using `generate.py`

> python generate.py --ckpt [CHECKPOINT PATH] IMG1 IMG2 IMG3 ...

## Samples

![Generated sample image](generated.png)
