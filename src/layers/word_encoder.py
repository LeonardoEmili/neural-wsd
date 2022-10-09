from transformers import AutoModel, AutoConfig, logging
import torch.nn as nn
import torch

from src.utils.torch_utilities import scatter_mean


class WordEncoder(nn.Module):
    def __init__(
        self,
        word_dropout: float = 0.1,
        word_projection_size: int = 512,
        fine_tune: bool = False,
        model_name: str = "bert-base-cased",
    ):
        super(WordEncoder, self).__init__()

        self.word_embedding = BertEmbedding(model_name=model_name, fine_tune=fine_tune)
        if "base" in model_name:
            word_embedding_size = 4 * 768
        else:
            word_embedding_size = 4 * 1024

        self.batch_normalization = nn.BatchNorm1d(word_embedding_size)
        self.output = nn.Linear(word_embedding_size, word_projection_size)
        self.word_dropout = nn.Dropout(word_dropout)

        # output size
        self.word_embedding_size = word_projection_size

    def forward(self, word_ids, subword_indices=None, sequence_lengths=None):
        word_embeddings = self.word_embedding(word_ids, sequence_lengths=sequence_lengths)

        # permute twice since batchnorm expects the temporal index on the last axis
        word_embeddings = word_embeddings.transpose(1, 2)
        word_embeddings = self.batch_normalization(word_embeddings)
        word_embeddings = word_embeddings.transpose(1, 2)

        word_embeddings = self.output(word_embeddings)
        # replace sigmoid with Swish
        word_embeddings = word_embeddings * torch.sigmoid(word_embeddings)
        word_embeddings = self.word_dropout(word_embeddings)

        # get word-level embeddings
        word_embeddings = scatter_mean(word_embeddings, subword_indices, dim=1)

        return word_embeddings


class BertEmbedding(nn.Module):
    """Wrapper of transformer's AutoModel class representing BERT word embedder."""

    def __init__(self, model_name="bert-base-cased", fine_tune=False):
        super(BertEmbedding, self).__init__()
        self.fine_tune = fine_tune
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        logging.set_verbosity_error()
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        logging.set_verbosity_warning()
        if not fine_tune:
            self.bert.eval()

    def forward(self, word_ids, sequence_lengths=None):
        timesteps = word_ids.shape[1]
        device = "cuda" if word_ids.get_device() == 0 else "cpu"
        # mask to avoid performing attention on padding token indices
        attention_mask = torch.arange(timesteps, device=device).unsqueeze(0) < sequence_lengths.unsqueeze(1)

        if not self.fine_tune:
            with torch.no_grad():
                # freeze bert's weights
                word_embeddings = self.bert(input_ids=word_ids, attention_mask=attention_mask)
        else:
            word_embeddings = self.bert(input_ids=word_ids, attention_mask=attention_mask)

        # concatenate the last four layers of BERT
        word_embeddings = torch.cat(word_embeddings[2][-4:], dim=-1)
        return word_embeddings
