import torch
print(torch.cuda.is_available())  # True if CUDA is available, False otherwise
print(torch.cuda.device_count())  # Number of available GPUs
print(torch.cuda.get_device_name(0))  # Name of the first GPU (if available)
print(torch.version.cuda)  # CUDA version used by PyTorch
