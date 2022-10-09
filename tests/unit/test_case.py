import unittest

from omegaconf import OmegaConf, DictConfig


class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conf: DictConfig = OmegaConf.load("tests/unit/root.yaml")
