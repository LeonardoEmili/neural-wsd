from omegaconf import OmegaConf

from src.readers.raganato_reader import SemCorReader, OMSTIReader, SemEvalReader
from tests.unit.test_case import TestCase


class RaganatoReaderTest(TestCase):
    def test_omsti(self):
        # samples = OMSTIReader.read(conf=self.conf, **self.conf.data.corpora["omsti"])
        # self.assertEqual(len(samples), 37176)
        pass

    def test_semcor_omsti(self):
        # samples = OMSTIReader.read(conf=self.conf, **self.conf.data.corpora["semcor+omsti"])
        # self.assertEqual(len(samples), 37176)
        pass

    def test_semcor(self):
        samples = SemCorReader.read(conf=self.conf, **self.conf.data.corpora["semcor"])
        self.assertEqual(len(samples), 37176)
        self.assertEqual(type(samples), list)

    def test_semeval_all(self):
        samples = SemEvalReader.read(conf=self.conf, **self.conf.data.corpora["semeval_all"], filter_ds="semeval_all")
        self.assertEqual(len(samples), 1173)
        self.assertEqual(type(samples), list)

    def test_semeval2007(self):
        samples = SemEvalReader.read(conf=self.conf, **self.conf.data.corpora["semeval_all"], filter_ds="semeval2007")
        self.assertEqual(len(samples), 135)
        self.assertEqual(type(samples), list)

    def test_semeval2013(self):
        samples = SemEvalReader.read(conf=self.conf, **self.conf.data.corpora["semeval_all"], filter_ds="semeval2013")
        self.assertEqual(len(samples), 306)
        self.assertEqual(type(samples), list)

    def test_semeval2015(self):
        samples = SemEvalReader.read(conf=self.conf, **self.conf.data.corpora["semeval_all"], filter_ds="semeval2015")
        self.assertEqual(len(samples), 138)
        self.assertEqual(type(samples), list)

    def test_senseval2(self):
        samples = SemEvalReader.read(conf=self.conf, **self.conf.data.corpora["semeval_all"], filter_ds="senseval2")
        self.assertEqual(len(samples), 242)
        self.assertEqual(type(samples), list)

    def test_senseval3(self):
        samples = SemEvalReader.read(conf=self.conf, **self.conf.data.corpora["semeval_all"], filter_ds="senseval3")
        self.assertEqual(len(samples), 352)
        self.assertEqual(type(samples), list)


if __name__ == "__main__":
    unittest.main()
