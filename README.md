
## Description   
This is my implementation of **Inpainting Transformer for Anomaly Detection** paper [https://arxiv.org/abs/2104.13897]
 
## How to run   
First, install dependencies   
```bash
# tested on Pytorch 1.9
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c 

conda install pytorch-lightning -c conda-forge

conda install piq -c photosynthesis-team -c conda-forge -c PyTorch
 ```   
 Next, run it.
 
 For **Training**   
 ```bash 
python training.py --img_root=./dataset/wood/  --gpus=2  --accelerator="ddp"  --batch_size=64  --max_epochs=500  
```
For **inference**
```bash

```

## Code Structure

The training and inference parts are separated into two files. 
1. training.py
2. inference.py (todo)

### Training
We use pytorch-lightning for organizing the code. 

Training starting point is `main()` method.


