from typing import *
import xml.etree.ElementTree as et
from omegaconf import DictConfig
from tqdm import tqdm
import hydra

class SemCorReader(object):
    data = []
    inst2wn = dict()

    @classmethod
    def read(cls, data_path: str, key_path : str, cached: Optional[Any] = None):
        if cached and isinstance(cached, list):
            cls.data = cached
        
        if not len(cls.data):
            cls._load_key(path=key_path)
            cls._load_xml(path=data_path)
        return cls.data

    @classmethod
    def _load_xml(cls, path: str, pattern: str = "./text/sentence"):
        """Parse the XML file and extract it into a data-driven format."""
        path = hydra.utils.to_absolute_path(path)
        root = et.parse(path).getroot()
        *_, name = path.rsplit("/", 1)
        iterator = tqdm(root.findall(pattern), desc=f"Reading XML dataset {name}")
        for id_sentence, sentence in enumerate(iterator):
            words = []
            instance_idxs = []
            instance_ids = []
            instance_senses = []
            instance_lemma_pos = []
            for j, word in enumerate(sentence):
                words.append(word.text)
                if word.tag == "instance":
                    # Store id, lemma, POS tags for instances
                    _id, _lemma, _pos = word.attrib["id"], word.attrib["lemma"], word.attrib["pos"]
                    instance_idxs.append(j)
                    instance_ids.append(_id)
                    instance_senses.append(cls.inst2wn[_id])
                    instance_lemma_pos.append(_lemma + "#" + _pos)

            cls.data.append(
                {
                    "sentence_id": id_sentence,
                    "sentence": words,
                    "index": instance_idxs,
                    "instance_id": instance_ids,
                    "sense": instance_senses,
                    "lexeme": instance_lemma_pos,
                }
            )

    @classmethod
    def _load_key(cls, path: str):
        """Extract the mapping to gold labels (i.e. WordNet ids)."""
        path = hydra.utils.to_absolute_path(path)
        with open(path, "r") as lines:
            for line in lines:
                instance_id, *wn_ids = line.rstrip().split()
                # an instance may have more than one gold annotation
                cls.inst2wn[instance_id] = wn_ids