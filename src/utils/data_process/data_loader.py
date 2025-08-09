from typing import List
from torch.utils.data import Dataset, DataLoader, random_split


class MyDataLoader:
    def __init__(self, my_dataset: Dataset, ratio: List[float], batch_size: int):
        self.my_dataset = my_dataset
        self.ratio = ratio
        self.batch_size = batch_size

    def prepare_data(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_data, validation_data, test_data = random_split(
            self.my_dataset, lengths=self.ratio
        )
        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        validation_loader = DataLoader(validation_data, batch_size=self.batch_size)
        test_loader = DataLoader(test_data, batch_size=self.batch_size)

        return train_loader, validation_loader, test_loader
