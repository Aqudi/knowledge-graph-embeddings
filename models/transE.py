from typing import Dict, Iterable
import torch
from torch import nn

from utils.types import TRIPLE


class TransE(nn.Module):
    def __init__(self, embedding_dim, margin, entity2id={}, relation2id={}):
        super().__init__()

    def forward(self, positive_tripls: Iterable[TRIPLE], negative_triples: Iterable[TRIPLE]):
        pass

    @staticmethod
    def create_negative_samples(triples:Iterable[TRIPLE], entity2id:Dict[str, int], device:str=None):
        size = (len(triples[0]),)

        # Negative sample을 만들 때 head를 바꿀지 tail을 바꿀지 선택
        target_entity_position = torch.randint(high=2, size=size, device=device)

        # 바꿀 random entity indices 생성
        random_entity_ids = torch.randint(low=0, high=len(entity2id), size=size, device=device)

        # Random entity로 교체
        heads, relations, tails =triples
        modified_heads = torch.where(target_entity_position==1, random_entity_ids, heads)
        modified_tails = torch.where(target_entity_position==0, random_entity_ids, tails)

        return [modified_heads, relations, modified_tails]
