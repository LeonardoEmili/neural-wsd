import os

from src.pl_modules import BasePLModule

from transformers import AutoTokenizer
from omegaconf import DictConfig
import streamlit as st
import tokenizers
import torch
import hydra
import spacy
import json


def load_spacy_tokenizer(model_name: str = "en_core_web_sm"):
    """Utility function to load SpaCy tokenizer."""
    return spacy.load(model_name)


@st.cache(allow_output_mutation=True)
def load_definitions(conf) -> dict:
    """Utility function to load WN definitions."""
    path = os.path.join(hydra.utils.to_absolute_path("."), conf.data.wordnet.glosses)
    with open(path, "r") as f_in:
        glosses_dict = json.load(f_in)
    return glosses_dict


def hash_config(config) -> str:
    """Overrides custom hashing for configuration object."""
    return config["_content"]["test"]["latest_checkpoint_path"]


@st.cache(hash_funcs={dict: hash_config}, allow_output_mutation=True)
def load_model(conf, n_classes: int, evaluation: bool = True) -> BasePLModule:
    """Utility function to load the pretrained model."""
    checkpoint_path = os.path.join(hydra.utils.to_absolute_path("."), "wsd_model.ckpt")
    model = BasePLModule.load_from_checkpoint(checkpoint_path, conf=conf, n_classes=n_classes)
    if evaluation:
        model = model.eval()
    return model


@st.cache(allow_output_mutation=True)
def load_vocabulary(conf: DictConfig) -> None:
    """Utility function to load the pretrained vocabulary."""
    base_path = os.path.join(hydra.utils.to_absolute_path("."), conf.data.preprocessed_dir, conf.data.train_ds)
    vocab_path = os.path.join(base_path, "vocab.pth")
    return torch.load(vocab_path)


def hash_tokenizer(obj) -> int:
    """Overrides custom hashing for tokenizer object."""
    return hash(obj.model)


@st.cache(hash_funcs={tokenizers.Tokenizer: hash_tokenizer}, allow_output_mutation=True)
def load_pretrained_tokenizer(tokenizer: str) -> AutoTokenizer:
    """Utility function to load the pretrained tokenizer."""
    return AutoTokenizer.from_pretrained(tokenizer)
