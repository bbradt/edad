import copy
import torch
from typing import Optional, Union


def n_bits(tensor):
    if tensor is None:
        return 0
    return 8 * tensor.nelement() * tensor.element_size()


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def get_average_model(surrogate_model, *args):
    surrogate_model.load_state_dict(args[0].state_dict())
    for model in args[1:]:
        state_dict = surrogate_model.state_dict()
        other_state_dict = model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = (v + other_state_dict[k]) / 2
        surrogate_model.load_state_dict(state_dict)
    return surrogate_model


def generate_original_PE(length: int, d_model: int) -> torch.Tensor:
    """Generate positional encoding as described in original paper.  :class:`torch.Tensor`

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos
        / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
    )
    PE[:, 1::2] = torch.cos(
        pos
        / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32) / d_model)
    )

    return PE


def generate_regular_PE(
    length: int, d_model: int, period: Optional[int] = 24
) -> torch.Tensor:
    """Generate positional encoding with a given period.

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.
    period:
        Size of the pattern to repeat.
        Default is 24.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))

    return PE


def generate_local_map_mask(
    chunk_size: int,
    attention_size: int,
    mask_future=False,
    device: torch.device = "cpu",
) -> torch.BoolTensor:
    """Compute attention mask as attention_size wide diagonal.

    Parameters
    ----------
    chunk_size:
        Time dimension size.
    attention_size:
        Number of backward elements to apply attention.
    device:
        torch device. Default is ``'cpu'``.

    Returns
    -------
        Mask as a boolean tensor.
    """
    local_map = np.empty((chunk_size, chunk_size))
    i, j = np.indices(local_map.shape)

    if mask_future:
        local_map[i, j] = (i - j > attention_size) ^ (j - i > 0)
    else:
        local_map[i, j] = np.abs(i - j) > attention_size

    return torch.BoolTensor(local_map).to(device)
