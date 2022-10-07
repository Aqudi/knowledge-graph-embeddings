import torch
from datasets.base import BaseDataset


class FB15k(BaseDataset):
    def __init__(self, filepath: str, entity2id, relation2id):
        super().__init__(entity2id, relation2id)
        self.data = self._read_file(filepath)
        self._init_item_to_index_map(self.data)

    def __getitem__(self, index):
        head, relation, tail = self.data.iloc[index]
        return self.entity2id[head], self.relation2id[relation], self.entity2id[tail]
