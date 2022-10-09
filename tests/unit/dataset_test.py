from transformers import AutoTokenizer

from src.dataset import SenseAnnotatedDataset
from tests.unit.test_case import TestCase


class DatasetTest(TestCase):
    def test_initialization(self):
        tokenizer = AutoTokenizer.from_pretrained(self.conf.model.tokenizer)
        dataset = SenseAnnotatedDataset(self.conf, tokenizer=tokenizer)
        self.assertIsNotNone(dataset)

    def test_loading_from_cache(self):
        tokenizer = AutoTokenizer.from_pretrained(self.conf.model.tokenizer)
        dataset = SenseAnnotatedDataset.from_cached(self.conf, tokenizer=tokenizer)
        self.assertIsNotNone(dataset)

    def test_cache_equivalence(self):
        tokenizer = AutoTokenizer.from_pretrained(self.conf.model.tokenizer)
        dataset = SenseAnnotatedDataset(self.conf, tokenizer=tokenizer)
        cached_dataset = SenseAnnotatedDataset.from_cached(self.conf, tokenizer=tokenizer)
        self.assertGreaterEqual(len(dataset.data), 0)
        self.assertEqual(len(dataset), len(cached_dataset))

    def test_bert_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.conf.model.tokenizer)
        dataset = SenseAnnotatedDataset.from_cached(self.conf, tokenizer=tokenizer)
        for sample, preprocessed_sample in zip(dataset.data, dataset.preprocessed_data):
            tokens = tokenizer(" ".join(sample["sentence"]), return_tensors="pt")
            self.assertIn("input_ids", tokens)
            self.assertIn("token_type_ids", tokens)
            self.assertIn("attention_mask", tokens)


if __name__ == "__main__":
    unittest.main()
