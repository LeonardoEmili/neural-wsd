from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from torch import nn
import torch

from src.utils.utilities import get_wordpiece_indices, batch_tokenizer, manual_training_step
from tests.unit.test_case import TestCase
from src.models.wsd_model import WSDModel


class ModelTest(TestCase):
    mock_samples = [
        (
            "BERT was created in 2018 by Jacob Devlin and his colleagues from Google.",
            torch.randint(0, 10, (10,)),  # dummy labels using 10 classes only
        ),
        (
            "Google Search consists of a series of localized websites.",
            torch.randint(0, 10, (10,)),  # dummy labels using 10 classes only
        ),
        (
            "Natural-language understanding is considered an AI-hard problem.",
            torch.randint(0, 10, (10,)),  # dummy labels using 10 classes only
        ),
    ]

    def test_parameters_change(self):
        n_classes = 10
        sentences, labels = zip(*self.mock_samples)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)

        tokenizer = AutoTokenizer.from_pretrained(self.conf.model.tokenizer)
        batch = batch_tokenizer(batch=sentences, tokenizer=tokenizer)

        # Define the model and store initial parameters
        model = WSDModel(self.conf, n_classes=n_classes)
        params = [param for param in model.base_parameters]
        initial_params = [param.clone() for param in model.base_parameters]

        # Set the model in `training` mode and update the weights
        manual_training_step(model=model, batch=batch, labels=labels)

        # Check if the weights are actually updated
        for p0, p1 in zip(initial_params, params):
            # using the more stable torch builtin function to check tensor equality
            assert not torch.equal(p0, p1)

    def test_output_range(self):
        n_classes = 10
        sentences, labels = zip(*self.mock_samples)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)

        tokenizer = AutoTokenizer.from_pretrained(self.conf.model.tokenizer)
        batch = batch_tokenizer(batch=sentences, tokenizer=tokenizer)

        # Define the model and store initial parameters
        model = WSDModel(self.conf, n_classes=n_classes)
        batch_out = manual_training_step(model=model, batch=batch, labels=labels)

        self.assertGreaterEqual(torch.min(batch_out["predictions"]), torch.tensor(0))
        self.assertLess(torch.max(batch_out["predictions"]), torch.tensor(n_classes))

    def test_nan_output(self):
        n_classes = 10
        sentences, labels = zip(*self.mock_samples)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)

        tokenizer = AutoTokenizer.from_pretrained(self.conf.model.tokenizer)
        batch = batch_tokenizer(batch=sentences, tokenizer=tokenizer)

        # Define the model and store initial parameters
        model = WSDModel(self.conf, n_classes=n_classes)
        batch_out = manual_training_step(model=model, batch=batch, labels=labels)

        self.assertFalse(batch_out["logits"].isnan().any())
        self.assertFalse(batch_out["predictions"].isnan().any())

    def test_inf_output(self):
        n_classes = 10
        sentences, labels = zip(*self.mock_samples)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)

        tokenizer = AutoTokenizer.from_pretrained(self.conf.model.tokenizer)
        batch = batch_tokenizer(batch=sentences, tokenizer=tokenizer)

        # Define the model and store initial parameters
        model = WSDModel(self.conf, n_classes=n_classes)
        batch_out = manual_training_step(model=model, batch=batch, labels=labels)

        self.assertTrue(batch_out["logits"].isfinite().all())
        self.assertTrue(batch_out["predictions"].isfinite().all())


if __name__ == "__main__":
    unittest.main()
