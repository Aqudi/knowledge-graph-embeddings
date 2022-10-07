from typing import Tuple
import pandas as pd
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base dataset with utility functions"""

    def __init__(self, entity2id={}, relation2id={}):
        super().__init__()
        self.data = None
        self.entity2id = entity2id
        self.relation2id = relation2id

    def __getitem__(self, index) -> Tuple[int, int, int]:
        raise NotImplementedError("You should implement __getitem__ method")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _read_file(filepath: str, delimiter="\t"):
        """Read file as pandas dataframe"""
        df = pd.read_csv(filepath, delimiter=delimiter, header=None)
        return df

    def _init_item_to_index_map(
        self, df: pd.DataFrame, entity_column_indexes=[0, 2], relation_column_index=1
    ):
        """Return index map dictionary of entities and relations in dataframe"""
        entities = pd.concat([df[df.columns[index]] for index in entity_column_indexes])
        relations = df[df.columns[relation_column_index]]

        if self.entity2id:
            entities = pd.concat([pd.Series(self.entity2id.keys()), entities])
        if self.relation2id:
            relations = pd.concat([pd.Series(self.relation2id.keys()), relations])

        uniq_entities = entities.unique()
        uniq_relation = relations.unique()

        self.entity2id.update({k: v for v, k in enumerate(uniq_entities)})
        self.relation2id.update({k: v for v, k in enumerate(uniq_relation)})
