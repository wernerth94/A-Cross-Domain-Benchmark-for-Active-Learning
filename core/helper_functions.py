from typing import Any, Optional, Union
import os
import torch
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def save_meta_data(logpath, agent, env, dataset):
    if not os.path.exists(logpath):
        os.makedirs(logpath, exist_ok=True)
    file = os.path.join(logpath, "meta.txt")
    if os.path.exists(file):
        os.remove(file)

    with open(file, "w") as f:
        if hasattr(dataset, "get_meta_data"):
            f.write("# Dataset: \n")
            f.write(f"{dataset.get_meta_data()} \n\n")
        if hasattr(agent, "get_meta_data"):
            f.write("# Agent: \n")
            f.write(f"{agent.get_meta_data()} \n\n")
        if hasattr(env, "get_meta_data"):
            f.write("# Environment: \n")
            f.write(f"{env.get_meta_data()} \n\n")


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



def plot_mean_std_development(inpt:list, title:str, out_file:str=None):
    # standard deviation statistics
    mean_develop = [np.mean(inpt[:i]) for i in range(1, len(inpt) + 1)]
    median_develop = [np.median(inpt[:i]) for i in range(1, len(inpt) + 1)]
    std_develop = [np.std(inpt[:i]) for i in range(1, len(inpt) + 1)]
    fig, ax = plt.subplots()
    ax.set_ylabel("mean")
    ax.set_xlabel("eval run")
    ax.grid()
    ax.scatter(range(len(inpt)), inpt, c="r", s=9, alpha=0.3)
    ax.plot(mean_develop, c="b", label="mean")
    ax.plot(median_develop, c="navy", label="median")
    ax2 = ax.twinx()
    ax2.set_ylabel("std")
    ax2.plot(std_develop, c="g", label="std")

    legend_elements = [
        Patch(facecolor="b", label="mean"),
        Patch(facecolor="navy", label="median"),
        Patch(facecolor="g", label="std"),
    ]
    ax2.legend(handles=legend_elements)
    ax.set_title(title)
    if out_file is None:
        plt.show()
    else:
        fig.savefig(out_file, dpi=100, bbox_inches='tight')
    plt.close(fig)