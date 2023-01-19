from typing import Any, Optional, Union
import torch
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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



def plot_mean_std_development(inpt:list, title:str, out_file:str):
    # standard deviation statistics
    mean_develop = [np.mean(inpt[:i]) for i in range(1, len(inpt) + 1)]
    std_develop = [np.std(inpt[:i]) for i in range(1, len(inpt) + 1)]
    fig, ax = plt.subplots()
    ax.set_ylabel("mean")
    ax.set_xlabel("eval run")
    ax.grid()
    ax.plot(mean_develop, c="b", label="mean")
    ax2 = ax.twinx()
    ax2.set_ylabel("std")
    ax2.plot(std_develop, c="g", label="std")

    legend_elements = [
        Patch(facecolor="b", label="mean"),
        Patch(facecolor="g", label="std"),
    ]
    ax2.legend(handles=legend_elements)
    ax.set_title(title)
    fig.savefig(out_file, dpi=100, bbox_inches='tight')