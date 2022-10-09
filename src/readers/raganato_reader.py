import re
import xml.etree.ElementTree as et
from typing import *

from omegaconf import DictConfig
from tqdm import tqdm
import hydra


class RaganatoReader(object):
    instance_keys = ("instance_indices", "instance_senses", "instance_lexemes", "instance_ids")

    @staticmethod
    def is_valid_cache(cached: Any) -> bool:
        """Validates the cache data."""
        return cached and isinstance(cached, list)

    @classmethod
    def read(
        cls,
        data_path: str,
        key_path: str,
        conf: DictConfig,
        cached: Optional[Any] = None,
        split: str = "train",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        cls.data = []
        cls.inst2wn = dict()

        if RaganatoReader.is_valid_cache(cached):
            cls.data = cached
            return cls.data

        cls._load_key(path=key_path, conf=conf)
        cls._load_xml(path=data_path, conf=conf, split=split, **kwargs)
        return cls.data

    @classmethod
    def _load_xml(
        cls,
        path: str,
        conf: DictConfig,
        pattern: str = "./text/sentence",
        split: str = "train",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Parse the XML file and extract it into a data-driven format."""
        path = hydra.utils.to_absolute_path(path)
        filter_ds = kwargs.get("filter_ds", None)
        if filter_ds:
            filter_ds = filter_ds.split("+")
        root = et.parse(path).getroot()
        *_, name = path.rsplit("/", 1)
        iterator = root.findall(pattern)
        total = len(iterator) if not conf.debug else min(len(iterator), conf.max_samples)
        for id_sentence, sentence in enumerate(tqdm(iterator, desc=f"Loading {split} dataset {name}", total=total)):
            corpus_id, _ = sentence.attrib["id"].split(".", 1)
            if conf.debug and len(cls.data) >= conf.max_samples:
                break
            if filter_ds and corpus_id not in filter_ds:
                # Filter out sentences from subcorpus (i.e. SemEval-ALL)
                continue
            words = []
            instance_idxs = []
            instance_ids = []
            instance_senses = []
            instance_lemma_pos = []
            for j, word in enumerate(sentence):
                words.append(word.text)
                if word.tag == "instance":
                    # Store id, lemma, POS tags for instances
                    _id, _lemma, _pos = (
                        word.attrib["id"],
                        word.attrib["lemma"],
                        word.attrib["pos"],
                    )
                    instance_idxs.append(j)
                    instance_ids.append(_id)
                    instance_senses.append(cls.inst2wn[_id])
                    instance_lemma_pos.append(_lemma + "#" + _pos)

            cls.data.append(
                {
                    "sentence_id": id_sentence,
                    "sentence": words,
                    "instance_indices": instance_idxs,  # indexes of senses
                    "instance_ids": instance_ids,  # "d000.s000.t000"
                    "instance_senses": instance_senses,
                    "instance_lexemes": instance_lemma_pos,
                }
            )

    @classmethod
    def _load_key(cls, path: str, conf: DictConfig):
        """Extract the mapping to gold labels (i.e. WordNet ids)."""
        path = hydra.utils.to_absolute_path(path)
        with open(path, "r") as lines:
            for line in lines:
                instance_id, *wn_ids = line.rstrip().split()
                # an instance may have more than one gold annotation
                cls.inst2wn[instance_id] = wn_ids if conf.data.allow_multiple_senses else wn_ids[0]


class SemCorReader(RaganatoReader):
    """RaganatoReader already implements all SemCorReader functionalities."""

    pass


class SemEvalReader(RaganatoReader):
    """RaganatoReader allows parsing the SemEval/Senseval datasets."""

    @classmethod
    def read(
        cls,
        data_path: str,
        key_path: str,
        conf: DictConfig,
        cached: Optional[Any] = None,
        split: str = "train",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """SemEvalReader read method requires a value for parameter [filter_ds]."""
        filter_ds = kwargs.get("filter_ds", None)
        assert filter_ds is not None, "Called SemEvalReader.read without passing a value for [filter_ds]"
        if "+" not in filter_ds:
            assert (
                filter_ds in conf.data.corpora
            ), f"{name} Dataset not supported (e.g. try with semcor, semeval2007, ...)"
            # Override data_path and key_path and filter the semeval_all dataset
            data_path = conf.data.corpora[filter_ds]["data_path"]
            key_path = conf.data.corpora[filter_ds]["key_path"]
            # No need to filter when using individual SemEval/Senseval datasets
            kwargs["filter_ds"] = None

        return super(SemEvalReader, cls).read(data_path, key_path, conf=conf, cached=cached, split=split, **kwargs)


class OMSTIReader(RaganatoReader):
    """OMSTIReader allows parsing the OMSTI dataset w/o performing the merge with SemCor."""

    @classmethod
    def read(
        cls,
        data_path: str,
        key_path: str,
        conf: DictConfig,
        cached: Optional[Any] = None,
        split: str = "train",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if RaganatoReader.is_valid_cache(cached):
            return super(OMSTIReader, cls).read(data_path, key_path, conf, cached, split, **kwargs)

        path_dict: Dict = OMSTIReader.preprocess_omsti(path=data_path)
        omsti_data = super(OMSTIReader, cls).read(path_dict["omsti"], key_path, conf=conf, split=split)
        if kwargs.get("merge_with_semcor", False):
            # Test with SemCor + OMSTI
            semcor_data = super(OMSTIReader, cls).read(path_dict["semcor"], key_path, conf=conf, split=split)
            for k, sample in enumerate(semcor_data):
                sample["sentence_id"] = len(omsti_data) + k
            omsti_data += semcor_data
        return omsti_data

    @staticmethod
    def preprocess_omsti(path: str) -> Dict[str, str]:
        """Splits the SemCor+OMSTI datasets into separate XML files."""
        path = hydra.utils.to_absolute_path(path)
        with open(path, "r") as f:
            omsti_corpus = f.readlines()

        # Split the original file looking at lines with <corpus ...> tag
        xml_header = omsti_corpus[0]
        idxs = [j for j, line in enumerate(omsti_corpus) if line.startswith("<corpus")]
        sources2path = {"semcor": "semcor", "mun": "omsti"}  # rename 'mun' -> 'omsti'
        # Extract corpus name from <corpus source="..."> tag
        sources = [re.findall(r"source=\"(.+?)\"", omsti_corpus[i])[0] for i in idxs]
        path, ext = path.rsplit(".", 1)
        output_paths = {sources2path[src]: f"{path}_{sources2path[src]}.{ext}" for src in sources}
        # Python ranges expect [i, j), adding the last index to consider the full doc
        if len(omsti_corpus) not in idxs:
            idxs.append(len(omsti_corpus))

        corpora = [xml_header + "".join(omsti_corpus[i:j]) for i, j in zip(idxs, idxs[1:])]
        for output_path, corpus in zip(output_paths.values(), corpora):
            with open(output_path, "w") as f:
                f.write(corpus)

        return output_paths
