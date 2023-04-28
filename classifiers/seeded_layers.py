import math
import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out, calculate_gain, _calculate_correct_fan

def kaiming_uniform_seeded(
    rng, tensor: torch.Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'
):

    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            kaiming_uniform_seeded,
            (tensor,),
            tensor=tensor,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity)

    if 0 in tensor.shape:
        print("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=rng)



class SeededLinear(nn.Linear):
    """
    Replaces the default kaiming initialization with a seeded version of itself
    """

    def __init__(self, model_rng, *args, **kwargs):
        self.model_rng = model_rng
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        kaiming_uniform_seeded(self.model_rng, self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            with torch.no_grad():
                self.bias.uniform_(-bound, bound, generator=self.model_rng)


class SeededConv2d(nn.Conv2d):
    """
    Replaces the default kaiming initialization with a seeded version of itself
    """

    def __init__(self, model_rng, *args, **kwargs):
        self.model_rng = model_rng
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        kaiming_uniform_seeded(self.model_rng, self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                self.bias.uniform_(-bound, bound, generator=self.model_rng)
                # init.uniform_(self.bias, -bound, bound)

