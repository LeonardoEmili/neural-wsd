"""
A simple script to scrape data from WordNet 3.1. database.
"""

import json
import os
from pathlib import Path

POS_TAGS = ["adj", "adv", "noun", "verb"]
WN_POS_TAGS = {"adj": "ADJ", "adv": "ADV", "noun": "NOUN", "verb": "VERB"}

WORDNET_URL = "https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz"
WORDNET_FILE = "wn3.1.dict.tar.gz"


def is_synset_id(token: str, synset_list: list[str]) -> bool:
    return token in synset_list


def write_json(dictionary: dict, path: str) -> None:
    with open(path, "w") as writer:
        json.dump(dictionary, writer, indent=4, sort_keys=True)


def is_header_line(line: str) -> bool:
    return line.startswith("  ")


def parse_glosses(path: str, output_dir: str) -> set:
    """Parses WordNet glosses and returns the set of synset ids."""
    synset_glosses = dict()
    for pos in POS_TAGS:
        with open(f"{path}data.{pos}", "r") as f_in:
            for line in f_in:
                if is_header_line(line):
                    continue
                left, gloss = line.strip().split("|", 1)
                synset_id, *_ = left.split(" ", 1)
                synset_glosses[synset_id] = gloss

    write_json(synset_glosses, path=f"{output_dir}/glosses.json")
    return set(synset_glosses.keys())


def parse_lexeme_means(path: str, synset_ids: set[str], output_dir: str) -> dict[str, list[str]]:
    """Builds the mapping lexeme -> possible synsets."""
    lexeme_means = dict()
    for pos in POS_TAGS:
        with open(f"{path}index.{pos}", "r") as f_in:
            for line in f_in:
                if is_header_line(line):
                    continue
                lemma, *line = line.split(" ")
                synsets = [token for token in line if is_synset_id(token, synset_ids)]
                lexeme_means[f"{lemma}#{WN_POS_TAGS[pos]}"] = synsets

    write_json(lexeme_means, path=f"{output_dir}/lexeme_means.json")
    return lexeme_means


def parse_lemma_means(lexeme_means: dict[str, list[str]], output_dir: str) -> None:
    """Builds the mapping lemma -> possible synsets."""
    lemma_means = dict()
    for lexeme, synsets in lexeme_means.items():
        lemma, pos = lexeme.split("#")
        if lemma not in lemma_means:
            lemma_means[lemma] = set()
        lemma_means[lemma].update(synsets)

    lemma_means = {k: list(v) for k, v in lemma_means.items()}
    write_json(lemma_means, path=f"{output_dir}/lemma_means.json")


def parse_sense_means(path: str, output_dir: str) -> None:
    sense_means = dict()
    with open(f"{path}index.sense", "r") as f_in:
        for line in f_in:
            sense, synset, *_ = line.strip().split(" ")
            sense_means[sense] = synset

    write_json(sense_means, path=f"{output_dir}/sense_means.json")


def main(path: str = "data/wordnet/", output_dir: str = "data/wordnet/means") -> None:

    # download wordnet's database from origin source
    if not os.path.exists(path):
        Path(path).mkdir(parents=True)
        os.system(f"curl {WORDNET_URL} -o {path}{WORDNET_FILE}")
        os.system(f"tar -xf {path}{WORDNET_FILE} -C {path}")
        os.system(f"rm {path}{WORDNET_FILE}")
        os.makedirs(f"{path}/means/", exist_ok=True)

    path += "dict/"

    # ... fetch glosses and the collection of available synsets
    synset_ids = parse_glosses(path, output_dir)

    # ... fetch the lexemes that are used by WordNet for indexing
    lexeme_means = parse_lexeme_means(path, synset_ids, output_dir)

    # ... extend the indexing to simple lemmas (i.e. dropping the POS tag)
    parse_lemma_means(lexeme_means, output_dir)

    # ... retrieve the mapping to go from senses to their synsets
    parse_sense_means(path, output_dir)


if __name__ == "__main__":
    main()
