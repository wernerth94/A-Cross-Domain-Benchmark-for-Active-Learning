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


class SeededEmbedding(nn.Embedding):
    def __init__(self, model_rng, *args, **kwargs):
        self.model_rng = model_rng
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.normal_(mean=0., std=1., generator=self.model_rng)
            self._fill_padding_idx_with_zero()

    @classmethod
    def from_pretrained(cls, model_rng, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
            freeze (bool, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
            padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                         therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                         i.e. it remains as a fixed "pad".
            max_norm (float, optional): See module initialization documentation.
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (bool, optional): See module initialization documentation. Default ``False``.
            sparse (bool, optional): See module initialization documentation.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            model_rng,
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            _freeze=freeze,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        return embedding


class SeededLSTM(nn.LSTM):

    def __init__(self, model_rng, *args, **kwargs):
        self.model_rng = model_rng
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
            for weight in self.parameters():
                weight.uniform_(-stdv, stdv, generator=self.model_rng)
