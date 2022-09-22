from typing import Dict, Iterable
import torch
from torch import nn
import numpy as np

from utils.types import TRIPLE


class TransE(nn.Module):
    def __init__(self, embedding_dim, margin, norm=1, device=None, entity2id={}, relation2id={}):
        super().__init__()

        # parameters
        self.embedding_dim=embedding_dim
        self.margin = margin
        self.norm = norm
        self.device = device
        self.entity2id=entity2id
        self.relation2id =relation2id

        # training setup
        self.criterior = nn.MarginRankingLoss(margin=margin)
        
        # layers
        self.relation_embedding = nn.Embedding(num_embeddings=len(self.relation2id), embedding_dim=embedding_dim, padding_idx=0)
        self.entity_embedding = nn.Embedding(num_embeddings=len(self.entity2id),embedding_dim=embedding_dim, padding_idx=0)

        # initialize embeddings
        initializer_factor = np.divide(6, np.sqrt(embedding_dim))
        self.entity_embedding.weight.data.uniform_(-initializer_factor, initializer_factor)
        self.relation_embedding.weight.data.uniform_(-initializer_factor, initializer_factor)

        # normalize rel_embedding
        self.relation_embedding.weight.data /= len(self.relation2id)

    def forward(self, positive_triples: Iterable[torch.Tensor], negative_triples: Iterable[torch.Tensor]):
        positive_distance = self._distance(positive_triples)
        negative_distance = self._distance(negative_triples)
        target = torch.tensor([-1], device=self.device)
        loss = self.criterior(positive_distance, negative_distance, target)
        return loss

    def normalize_entity_embedding(self):
        self.entity_embedding.weight.data /= len(self.entity2id)

    def _distance(self, triples:Iterable[torch.Tensor]):
        head, relation, tail = triples
        
        # embedding layer
        head_embedding = self.entity_embedding(head)
        relation_embedding = self.relation_embedding(relation)
        tail_embedding = self.entity_embedding(tail)
        
        # calculate norm
        result = (head_embedding+relation_embedding) - tail_embedding
        norm = result.norm(p=self.norm, dim=1)
        print(norm)
        return norm

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

if __name__ == "__main__":
    from datasets.FB15k import FB15k
    from models.transE import TransE
    from torch.utils.data import DataLoader

    data = FB15k("data/FB15k/freebase_mtr100_mte100-test.txt")

    model = TransE(
        embedding_dim=10,
        margin=5,
        entity2id=data.entity2id,
        relation2id=data.relation2id,
    )

    train_dataloader = DataLoader(data, batch_size=3)

    positive_triples = next(iter(train_dataloader))
    negative_triples = TransE.create_negative_samples(positive_triples, data.entity2id)
    
    loss = model(positive_triples, negative_triples)
    print(loss)
    


