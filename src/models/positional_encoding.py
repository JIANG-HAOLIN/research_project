import torch
import math


class StandardPositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int = 256, max_len: int = 5000) -> None:
        """Add positional encoding from tutorial 6 to the input tokens for transformer.

        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """

        Args:
            x: input for positional encoding [ B, N, D]
                N: sequence length B: batch size D: length of token

        Returns:
             output for positional encoding [ B, N, D]
                N: sequence length B: batch size D: length of token

        """
        x = x + self.pe[:, :x.size(1)]
        return x


