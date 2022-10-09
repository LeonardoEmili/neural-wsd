from collections import Counter, defaultdict
from operator import itemgetter
from dotenv import load_dotenv
from typing import *
import wandb
import os

from omegaconf import DictConfig, OmegaConf, open_dict
from transformers.tokenization_utils_base import BatchEncoding
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from torch.optim import Optimizer
from torchtext.vocab import Vocab
import torch.nn.functional as F
import torch
import hydra
import yaml
import json


def read_json_hydra(path: str) -> any:
    base_path = hydra.utils.to_absolute_path(".")
    path = os.path.join(base_path, path)
    with open(path, "r") as f_in:
        result = json.load(f_in)
    return result


def manual_training_step(
    model: torch.nn.Module,
    batch: BatchEncoding,
    labels: torch.Tensor,
    optimizer: Optimizer = torch.optim.Adam,
    loss_fn: F.cross_entropy = F.cross_entropy,
) -> None:
    """Performs a training step of the input `model` using a custom training configuration."""
    optim = optimizer(model.parameters())
    optim.zero_grad()
    logits = model(batch)
    loss = loss_fn(logits, labels)
    loss.backward()
    optim.step()
    return {"logits": logits, "predictions": logits.argmax(-1)}


def get_batch_wordpiece_indices(
    batch: Union[list[list[str]], list[str]],
    tokenizer: AutoTokenizer,
    padding: int = 0,
    return_tensors: Optional[str] = None,
) -> list[Union[torch.tensor, list[int]]]:
    transform_fn = lambda x: torch.tensor(x) if return_tensors == "pt" else x
    return [transform_fn(get_wordpiece_indices(sentence, tokenizer)) for sentence in batch]


def get_wordpiece_indices(
    sentence: list[str],
    tokenizer: AutoTokenizer,
) -> Union[torch.tensor, list[int]]:
    indices = []
    for idx_word, word in enumerate(sentence):
        word_tokenized = tokenizer.tokenize(word)
        for _ in range(len(word_tokenized)):
            indices.append(idx_word)
    return indices


def batch_tokenizer(
    batch: Union[list[list[str]], list[str]],
    tokenizer: AutoTokenizer,
    padding: int = 0,
) -> BatchEncoding:
    assert len(batch) > 0
    if isinstance(batch[0], str):
        # The input batch should be already tokenized into words
        batch = [sentence.split(" ") for sentence in batch]
    lengths = torch.tensor([len(sentence) for sentence in batch])
    word_pieces_indexes = get_batch_wordpiece_indices(batch, tokenizer, padding=padding, return_tensors="pt")
    batch = tokenizer(batch, return_tensors="pt", is_split_into_words=True, padding=True, add_special_tokens=False)
    batch["lengths"] = lengths
    batch["word_pieces_indexes"] = pad_sequence(word_pieces_indexes, batch_first=True, padding_value=padding)
    return batch


def sentence_tokenizer(sentence: list[str], tokenizer: AutoTokenizer) -> tuple[dict[str, torch.Tensor], list]:
    """Returns the input sentence after applying tokenization."""
    sentence_tokenized = tokenizer(" ".join(sentence), return_tensors="pt", add_special_tokens=False)
    indexes: list[int] = get_wordpiece_indices(sentence, tokenizer=tokenizer)
    return {k: v[0] for k, v in sentence_tokenized.items()}, indexes


def wandb_login() -> None:
    """Weights and Biases login using environmental key."""
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_KEY"))


def get_checkpoint_path(conf: DictConfig) -> str:
    """Returns the model checkpoint path from the config file."""
    checkpoint_path = conf.test.checkpoint_path
    if conf.test.use_latest and "latest_checkpoint_path" in conf.test:
        checkpoint_path = conf.test.latest_checkpoint_path
    return hydra.utils.to_absolute_path(checkpoint_path)


def update_latest_checkpoint_path(model_path: str, config_path: str = "conf/test/default_test.yaml") -> None:
    """Useful function that updates the field ``best_model_path`` when the training is complete."""
    base_path = hydra.utils.to_absolute_path(".")
    config_path = os.path.join(base_path, config_path)
    with open(config_path, "r") as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            yaml_dict["latest_checkpoint_path"] = os.path.relpath(model_path, start=base_path)
        except Exception as e:
            print(e)

    with open(config_path, "w") as stream:
        yaml.dump(yaml_dict, stream)


def gpus(conf: DictConfig) -> int:
    """Utility to determine the number of GPUs to use."""
    return conf.train.pl_trainer.gpus if torch.cuda.is_available() else 0


def add_configuration_field(conf: DictConfig, field: str, value: Any) -> None:
    """
    Adds a new struct flag.
    Docs: https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#struct-flag
    """
    OmegaConf.set_struct(conf, True)
    with open_dict(conf):
        conf[field] = value


def vocab_lookup_indices(vocabulary: Union[Vocab, defaultdict], tokens: List[str]) -> List[int]:
    """Replacement for Vocab's method ``lookup_indices`` introduced in latest version of torchtext."""
    return [vocabulary[token] for token in tokens]


def collate_fn(
    batch: List[Tuple[Dict[str, Any]]],
    batch_keys: Tuple[str],
    lemma_means: Optional[defaultdict] = None,
    lexeme_means: Optional[defaultdict] = None,
    output_dim: Optional[int] = None,
    padding_value: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    A simple collate function used to provide batched input to our models.
    :param batch: a zipped list of tuples containing dataset values
    :param batch_keys: a tuple of keys describing the values in the batch
    :param padding_value: the value to use to pad input sequences
    :return a batch of preprocessed input sentences
    """
    # Unroll the list of tuples into a more useful dictionary with batched features
    batch = dict(zip(batch_keys, zip(*[itemgetter(*batch_keys)(x) for x in batch])))
    word_pieces_indexes: List[torch.Tensor] = [torch.tensor(elem) for elem in batch["word_pieces_indexes"]]
    # Create the output labels with sense identifiers at position senses_indices
    senses_ = torch.zeros(len(batch["senses_indices"]), max(batch["lengths"]), dtype=torch.long)
    for batch_idx, (_indices, _senses) in enumerate(zip(batch["senses_indices"], batch["senses"])):
        senses_[batch_idx, _indices] = torch.tensor(_senses, dtype=torch.long)

    assert not (lemma_means and lexeme_means), "Specify either one among use_lemma_mask OR use_lexeme_mask"

    if lemma_means or lexeme_means:
        assert output_dim, "Output dimension is required to create the ``sense_mask``"
        if lemma_means:
            batch["sense_mask"] = create_senses_mask(batch, "lemmas", lemma_means, senses_.shape, output_dim)
        else:
            batch["sense_mask"] = create_senses_mask(batch, "lexemes", lexeme_means, senses_.shape, output_dim)

    batch["input_ids"] = pad_sequence(batch["input_ids"], batch_first=True, padding_value=padding_value)
    batch["attention_mask"] = pad_sequence(batch["attention_mask"], batch_first=True, padding_value=padding_value)
    batch["word_pieces_indexes"] = pad_sequence(word_pieces_indexes, batch_first=True, padding_value=padding_value)
    batch["senses"] = senses_
    batch["lengths"] = torch.tensor(batch["lengths"])

    return batch


def create_senses_mask(
    batch: list,
    field: str,
    means: defaultdict,
    senses_dim: tuple[int, int],
    output_dim: int,
) -> torch.Tensor:
    """Computes the sense mask from the provided ``means`` to limit predictions to candidate senses only."""
    batch_sense_mask_idxs = [vocab_lookup_indices(means, tokens) for tokens in batch[field]]
    batch_sense_mask = torch.ones((*senses_dim, output_dim), dtype=torch.bool)
    for batch_idx, (sense_idxs, sense_mask_idxs) in enumerate(zip(batch["senses_indices"], batch_sense_mask_idxs)):
        batch_sense_mask[batch_idx, sense_idxs] = False
        for sense_idx, _sense_mask_idxs in zip(sense_idxs, sense_mask_idxs):
            batch_sense_mask[batch_idx, sense_idx, _sense_mask_idxs] = True
    return batch_sense_mask
