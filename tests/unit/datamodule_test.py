from src.pl_data_modules import BasePLDataModule
from tests.unit.test_case import TestCase


class DataModuleTest(TestCase):
    def test_initialization(self):
        pl_data_module = BasePLDataModule(self.conf)
        self.assertIsNotNone(pl_data_module)

    def test_batches_num(self):
        pl_data_module = BasePLDataModule(self.conf)
        batch_size = self.conf.data.batch_size

        self.assertEqual(
            (len(pl_data_module.train_dataset) + batch_size - 1) // batch_size,
            len(
                pl_data_module.train_dataloader(),
            ),
        )
        self.assertEqual(
            (len(pl_data_module.valid_dataset) + batch_size - 1) // batch_size,
            len(
                pl_data_module.val_dataloader(),
            ),
        )
        self.assertEqual(
            (len(pl_data_module.test_dataset) + batch_size - 1) // batch_size,
            len(
                pl_data_module.test_dataloader(),
            ),
        )


if __name__ == "__main__":
    unittest.main()
