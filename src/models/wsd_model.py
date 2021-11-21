from hydra.utils import instantiate
from omegaconf import DictConfig
import torch.nn as nn
import torch


class WSDModel(nn.Module):
    def __init__(self, conf: DictConfig, n_classes: int):
        super().__init__()
        self.conf = conf
        self.n_classes = n_classes
        self.word_encoder = instantiate(conf.model.word_encoder)

        if conf.model.sequence_encoder == "lstm":
            self.sequence_encoder = instantiate(conf.model.lstm_encoder)
            self.hidden_size = conf.model.lstm_encoder.hidden_size * conf.model.lstm_encoder.num_layers
        else:
            self.sequence_encoder = IdentityLayer()
            self.hidden_size = self.word_encoder.word_embedding_size

        self.output_layer = torch.nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, x: dict) -> torch.Tensor:
        word_ids, subword_indices, lengths = x["input_ids"], x["word_pieces_indexes"], x["lengths"]
        result = self.word_encoder(word_ids, subword_indices=subword_indices, sequence_lengths=lengths)
        sequence_out, _ = self.sequence_encoder(result)  # batch, seq_len, hidden state
        return self.output_layer(sequence_out)


class IdentityLayer(nn.Module):
    """Syntactic sugar to simplify the sequence encoder"""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, None
