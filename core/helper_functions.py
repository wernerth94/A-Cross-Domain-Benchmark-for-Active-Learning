from typing import Any, Optional, Union
import torch
import numpy as np

def to_torch(x: Any, dtype: Optional[torch.dtype] = None,
             device: Union[str, int, torch.device] = "cpu", ) -> torch.Tensor:
    """
    Convert an object to torch.Tensor
    Ref: Tianshou
    """
    if isinstance(x, np.ndarray) and issubclass(
        x.dtype.type, (np.bool_, np.number)
    ):  # most often case
        x = torch.from_numpy(x).to(device)
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)
    else:  # fallback
        raise TypeError(f"object {x} cannot be converted to torch.")