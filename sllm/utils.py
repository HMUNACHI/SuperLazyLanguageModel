import gc
import math
import os
import random
import shutil

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from sllm.common import DTYPE, GRADIENT_DIR


def load_tensor_from_storage(weight_path, shape, dtype=DTYPE, to_ram=False):
    """
    Load a tensor from a binary file.

    This function reads raw data from a file using torch.from_file and reshapes it into the desired tensor shape.
    If the flag 'to_ram' is set to True, the tensor is cloned into RAM to avoid memory-mapping.

    Args:
        weight_path (str): The file path to the binary weight file.
        shape (tuple or int): Desired shape of the tensor. If a tuple is provided, the total number 
            of elements is computed as the product of the tuple components; otherwise, the shape is treated as the total size.
        dtype (torch.dtype, optional): Data type of the tensor. Defaults to DTYPE.
        to_ram (bool, optional): If True, clones the tensor to ensure it resides in RAM. Defaults to False.

    Returns:
        torch.Tensor: The tensor with the specified shape loaded from the file.
    """
    if isinstance(shape, tuple):
        size = math.prod(shape)
    else:
        size = shape

    data = torch.from_file(weight_path, shared=False, size=size, dtype=dtype)
    data = data.view(shape)

    if to_ram:
        data = data.clone()

    return data


@torch._dynamo.disable
def save_tensor_to_storage(weight_path, data):
    """
    Save a tensor to a binary file.

    The function first ensures the tensor is contiguous and detached from the computation graph.
    It then converts the tensor to a NumPy array and writes it to the specified file in binary format.

    Args:
        weight_path (str): The file path where the tensor will be saved.
        data (torch.Tensor): The tensor to be saved.

    Returns:
        None
    """
    data.contiguous().detach().numpy().tofile(weight_path)


def download_weights(weight_dir, model_name):
    """
    Download and save the pretrained model weights to a local directory.

    This function checks if the local weight directory already exists. If not, it creates the directory,
    loads the pretrained model using Hugging Face's AutoModelForCausalLM with the specified torch dtype,
    and saves each model parameter as a binary file in the directory. After saving, the model is deleted
    and garbage is collected to free memory.

    Args:
        weight_dir (str): The local directory path where weights will be stored.
        model_name (str): The name or identifier of the pretrained model to download.

    Returns:
        None
    """
    if os.path.exists(weight_dir):
        return

    os.makedirs(weight_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE)

    for name, param in model.named_parameters():
        file_path = f"{weight_dir}/{name}.bin"
        param.detach().numpy().tofile(file_path)

    del model
    gc.collect()


def remove_weights(weight_dir):
    """
    Remove the weights stored in a specified directory or file.

    If the provided path is a directory, the entire directory is deleted recursively.
    Otherwise, if it is a file, the file is removed.

    Args:
        weight_dir (str): The file or directory path where the weights are stored.

    Returns:
        None
    """
    if os.path.exists(weight_dir):
        if os.path.isdir(weight_dir):
            shutil.rmtree(weight_dir)
        else:
            os.remove(weight_dir)


def clear_gradient_dir():
    """
    Clear the gradient directory.

    This function removes the entire gradient directory (if it exists) and then creates an empty directory with the same path.
    This is used to manage temporary storage during gradient computations.

    Returns:
        None
    """
    if os.path.exists(GRADIENT_DIR):
        shutil.rmtree(GRADIENT_DIR)
    os.makedirs(GRADIENT_DIR, exist_ok=True)
