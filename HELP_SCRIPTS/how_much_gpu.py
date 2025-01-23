import torch
print("Torch CUDA Available:", torch.cuda.is_available())
print("Torch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())
print("How much cache is empty:", torch.cuda.empty_cache())

# Free GPU memory
torch.cuda.empty_cache()
torch.cuda.synchronize()
