from typing import Callable
import torch
import torch.nn.functional as F


class EmbeddingPooler:
    """
    This class hosts various methods for obtaining a batch of token-level embeddings
    to a sentence level embedding. Notice that using Sentence-Bert directly didn't work.

    All function take as input an array of dimensions (batch_size, max_seq_len, hidden_dim),
    where max_seq_len is dependent upon the examples in the batch (dynamic), and
    hidden_dim is the number of hidden units that's model dependent

    Max pooling?
    """

    def __init__(self):
        self.name_to_fct = {
            "mean_with_attention_heads": self.mean_with_attention_heads,
            "mean_with_attention": self.mean_with_attention,
            "mean": self.mean,
            "cls": self.cls,
            "flatten": self.flatten,
            "eos": self.eos,
            "last": self.last,
        }

    def mean(self, hidden_states: torch.tensor, *_) -> torch.tensor:
        return torch.mean(hidden_states, dim=1).squeeze().detach().cpu()

    def mean_with_attention(
        self, hidden_states: torch.tensor, attention_mask: torch.tensor
    ) -> torch.tensor:
        # NOTE: this attention is actually just the input mask telling
        # the model to ignore padded tokens for batched inputs
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )
        return (
            (
                torch.sum(hidden_states * input_mask_expanded, 1)
                / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            )
            .squeeze()
            .detach()
            .cpu()
        )

    def last(self, hidden_states: torch.tensor, attention_mask: torch.tensor):
        """
        Selects the hidden states of the last token (that is not a pad token),
        in the array of hidden_states. Based on the provided mask of non-pad
        tokens.

        Args:
            hidden_states (torch.tensor): (batch_size, seq_length, hidden_size)
            attention_mask (torch.tensor): (batch_size, seq_length)

        Returns:
            torch.tensor: (batch_size, hidden_size)
        """
        # Compute the lengths of sequences (number of non-pad tokens per sequence)
        seq_lengths = (
            attention_mask.sum(dim=1) - 1
        )  # subtracting 1 to get the index of the last non-pad token

        # Gather the last non-pad token's hidden state for each sequence
        batch_size = hidden_states.shape[0]
        batch_indices = torch.arange(batch_size)
        last_hidden_states = hidden_states[batch_indices, seq_lengths]

        return last_hidden_states.detach().cpu()

    def mean_with_attention_heads(
        self, hidden_states: torch.tensor, attention: torch.tensor
    ) -> torch.tensor:
        """
        Average over the attention heads/weights per token, and then
        compute the representation as attention \dot hidden_state

        Args:
            hidden_states (torch.tensor): (batch_size, seq_length, hidden_size)
            attention (torch.tensor): (batch_size, num_heads, seq_length, seq_length)

        Returns:
            torch.tensor: (batch_size, hidden_size)
        """

        # Average attention weights across heads
        att_avg_head = attention.mean(
            dim=1
        )  # (batch_size, seq_length, seq_length)

        # Average attention weights across tokens (for simplicity, you might want more sophisticated methods)
        att_avg_tokens = att_avg_head.mean(dim=-1)  # (batch_size, seq_length)

        # Apply the attention weights to the hidden states
        return (hidden_states * att_avg_tokens.unsqueeze(-1)).sum(dim=1)

    def cls(self, hidden_states: torch.tensor, *_):
        return (
            hidden_states[:, 0, :].squeeze().detach().cpu()
        )  # first token is always cls

    def eos(self, hidden_states: torch.tensor, attention_mask: torch.tensor):
        # return the last valid token
        hs = []
        for h, m in zip(hidden_states, attention_mask):
            idx = ((m == 1).nonzero(as_tuple=True)[-1])[-1]
            hs.append(h[idx].squeeze().detach().cpu())
        res = torch.vstack(hs).detach().cpu()
        return res

    def flatten(self, hidden_states: torch.tensor, *_):
        return hidden_states.reshape(hidden_states.shape[0], -1).detach().cpu()

    def get(self, name: str) -> Callable[[torch.tensor], torch.tensor]:
        return self.name_to_fct[name]
