import numpy as np
import torch as th
from torch.autograd import Variable


def identity(x):
    return x


def index_to_one_hot(index, dim):
    if isinstance(index, (int, np.integer)):
        one_hot = np.zeros(dim, dtype=np.float32)
        one_hot[index] = 1.0
    else:
        one_hot = np.zeros((len(index), dim), dtype=np.float32)
        one_hot[np.arange(len(index)), index] = 1.0
    return one_hot


def to_tensor_var(x, use_cuda=True, dtype="float"):
    if dtype == "float":
        tensor = th.tensor(x, dtype=th.float32)
    elif dtype == "long":
        tensor = th.tensor(x, dtype=th.long)
    elif dtype == "byte":
        tensor = th.tensor(x, dtype=th.uint8)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if use_cuda and th.cuda.is_available():
        tensor = tensor.cuda()

    return Variable(tensor)


def agg_double_list(l):
    s = [np.array(np.sum(np.array(x), axis=0)) for x in l]
    s_mu = np.mean(np.array(s), axis=0)
    s_std = np.std(np.array(s), axis=0)
    return s_mu, s_std