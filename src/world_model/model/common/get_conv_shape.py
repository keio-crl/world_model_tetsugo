import numpy as np
from typing import Tuple, Iterable


def get_conved_size(
    obs_shape: Tuple[int, int],
    channels: Tuple[int, ...],
    kernels: Tuple[int, ...],
    strides: Tuple[int, ...],
    paddings: Tuple[int, ...],
):
    conved_shape = obs_shape

    for i in range(len(channels)):
        conved_shape = conv_out_shape(conved_shape, paddings[i], kernels[i], strides[i])
    conved_size = channels[-1] * np.prod(conved_shape).item()
    return conved_size


def conv_out_shape(h_in: Iterable, padding: int, kernel_size: int, stride: int):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def conv_out(h_in: int, padding: int, kernel_size: int, stride: int):
    return int((h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0)
