from typing import Tuple
import pandas as pd
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """Base dataset with utility functions"""
    def __init__(self):
        super().__init__()
        self.data = None
        self.entity2id = None
        self.relation2id = None

    def __getitem__(self, index) -> Tuple[int, int, int]:
        raise NotImplementedError("You should implement __getitem__ method")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _read_file(filepath:str, delimiter="\t"):
        """Read file as pandas dataframe"""
        df = pd.read_csv(filepath, delimiter=delimiter, header=None)
        return df

    @staticmethod
    def _get_item_to_index_map(df:pd.DataFrame, entity_column_indexes=[0,2], relation_column_index=1):
        """Return index map dictionary of entities and relations in dataframe"""
        entities = pd.concat([df[df.columns[index]]for index in entity_column_indexes])
        relation =df[df.columns[relation_column_index]]

        uniq_entities = entities.unique()
        uniq_relation = relation.unique()

        entity2id = {k:v for v, k in enumerate(uniq_entities)}
        relation2id = {k:v for v, k in enumerate(uniq_relation)}

        return [entity2id,relation2id]
