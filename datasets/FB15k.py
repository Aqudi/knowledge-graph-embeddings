from typing import Tuple
from datasets.base import BaseDataset

class FB15k(BaseDataset):
    def __init__(self, filepath: str):
        super().__init__()
        self.data = self._read_file(filepath)
        self.entity2id, self.relation2id = self._get_item_to_index_map(self.data)

    def __getitem__(self, index) -> Tuple[int, int, int]:
        return tuple(self.data.iloc[index])