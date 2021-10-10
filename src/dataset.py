from typing import *
from torch.utils.data import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer
from src.readers.semcor_reader import SemCorReader


class SenseAnnotatedDataset(Dataset):
    def __init__(
        self,
        conf: DictConfig,
        tokenizer: AutoTokenizer,
        name: str = 'semcor',
        cached: Optional[Any] = None
    ) -> None:
        assert name in ['semcor', 'semcor+omsti']
        self.conf = conf
        self.name = name
        self.tokenizer = tokenizer
        if name == 'semcor':
            self.data = SemCorReader.read(
                data_path=conf.data.semcor_data_path,
                key_path=conf.data.semcor_key_path,
                cached=cached
            )

    def __getitem__(self, idx: int):
        sentence = self.data[idx]['sentence']
        sentence_tokenized, indexes = self.bert_tokenizer(sentence)
        self.data[idx]['bert_sentence'] = sentence_tokenized
        self.data[idx]['word_pieces_indexes'] = indexes
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def indices_word_pieces(self, sentence: List[str]) -> List[int]:
        indices = []
        for idx_word, word in enumerate(sentence):
            word_tokenized = self.tokenizer.tokenize(word)
            for _ in range(len(word_tokenized)):
                indices.append(idx_word)
        return indices

    def bert_tokenizer(self, sentence: List[str]):
        sentence_tokenized = self.tokenizer(
            " ".join(sentence), return_tensors="pt")
        indexes: List[int] = self.indices_word_pieces(sentence)
        return sentence_tokenized, indexes

    def dump_data(self, path: str) -> None:
        if path.endswith('.pth'):
            torch.save(self.data, output_path)
        else:
            result = {entry["sentence_id"]: entry for entry in self.data}
            with open(path, "w") as writer:
                json.dump(result, writer, indent=4, sort_keys=True)

    def load_data(self, path: str) -> Dict:
        with open(path, "r") as reader:
            return json.load(reader)

    @classmethod
    def from_cached(cls, conf: DictConfig, path):
        *path, name, ext = re.split("/|\.", path)
        if path.endswith('.pth'):
            data: List = torch.load(path)
            return cls(conf=conf, name=name, cached=data)

        assert not path.endswith('.json'), "Extension not supported"
        data: dict = load_data()
        return cls(conf=conf, name=name, cached=list(data.values()))
