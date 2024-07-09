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
            "mean_with_attention": self.mean_with_attention,
            "mean": self.mean,
            "cls": self.cls,
            "flatten": self.flatten,
            "eos": self.eos
        }

    def mean(self, hidden_states: torch.tensor, *_) -> torch.tensor:
        return torch.mean(hidden_states, dim=1).squeeze().detach().cpu()

    def mean_with_attention(self, hidden_states: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        return (torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)).squeeze().detach().cpu()

    def cls(self, hidden_states: torch.tensor, *_):
        return hidden_states[:, 0, :].squeeze().detach().cpu() # first token is always cls

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
