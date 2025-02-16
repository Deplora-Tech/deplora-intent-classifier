import torch
import bitsandbytes as bnb

print("PyTorch CUDA Available:", torch.cuda.is_available())
print("BitsAndBytes Version:", bnb.__version__)
