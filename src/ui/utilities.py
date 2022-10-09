from src.utils.utilities import sentence_tokenizer
from src.models.wsd_model import WSDModel

from transformers import AutoTokenizer
from torchtext.vocab import Vocab
from torch.nn import Softmax
import streamlit as st
import torch
import spacy


def preprocess_sentence(
    sentence: str,
    bert_tokenizer: AutoTokenizer,
    word_tokenizer: spacy.Language,
    sentence_id: int = 42,
) -> dict:
    """Prepares the input sentences by applying word-level and subword-level tokenization."""
    tokens = word_tokenizer(sentence)
    spans = [token.text for token in tokens]
    sentence_tokenized, indexes = sentence_tokenizer(spans, bert_tokenizer)

    return {
        **sentence_tokenized,
        "lengths": len(tokens),
        "word_pieces_indexes": indexes,
        "sentence_id": sentence_id,
        "senses_indices": list(range(len(tokens))),
        "senses": [-1 for _ in range(len(tokens))],
        "lexemes": [f"{token.lemma_}#{token.pos_}" for token in tokens],
        "lemmas": [token.lemma_ for token in tokens],
        "tokens": tokens,
    }


@torch.no_grad()
def predict(model: WSDModel, batch: dict, use_lexeme_mask: bool = True) -> torch.Tensor:
    """Runs a forward pass to get model's predictions."""
    output = dict()
    softmax = Softmax(dim=1)

    output["logits"] = model.model(batch)
    if use_lexeme_mask and "sense_mask" in batch:
        output["logits"].masked_fill_(~batch["sense_mask"], float("-inf"))

    output["logits"] = output["logits"].squeeze()

    output["scores"] = softmax(output["logits"])
    return output


def get_disambiguated_tokens(sentence: dict, predictions: dict, output_vocabulary: Vocab, threshold: int) -> list:
    """Returns the list of tokens whose score is higher than the provided threshold."""
    disambiguated_tokens = []
    for idx, (token, output_distribution) in enumerate(zip(sentence["tokens"], predictions["scores"])):
        index = torch.argmax(output_distribution)
        score = output_distribution[index].item()
        if score >= threshold:
            disambiguated_tokens.append((idx, token, score, output_vocabulary.itos[index]))

    return disambiguated_tokens
