import numpy as np
import os
import random
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_free_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmax(memory_available)


def get_device():
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{get_free_gpu()}")
    else:
        device = torch.device("cpu")
    return device
