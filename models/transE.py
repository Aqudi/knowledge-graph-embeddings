from argparse import ArgumentParser
from typing import Dict, Iterable, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

import pytorch_lightning as pl

from utils import metrics

class TransEDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_dataset, test_dataset, val_dataset):
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataset", "test_dataset", "val_dataset"])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.hparams.batch_size)

class TransE(pl.LightningModule):
    def __init__(self, embedding_dim, margin, lr, norm, entity2id={}, relation2id={}, *args, **kwargs):
        super().__init__()

        # parameters
        self.embedding_dim=embedding_dim
        self.margin = margin
        self.lr = lr
        self.norm = norm
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

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("TransE")
        parser.add_argument("--embedding_dim", default=20)
        parser.add_argument("--lr", default=0.001)
        parser.add_argument("--margin", default=1)
        parser.add_argument("--norm", default=1)

    def forward(self, x):
        """Inference only"""
        return self._distance(x)

    def training_step(self, batch: Iterable[Tuple[int, int, int]], _):
        """Training"""
        negative_triples = self.create_negative_samples(batch, self.entity2id)

        positive_distance = self.forward(batch)
        negative_distance = self.forward(negative_triples)
        target = torch.tensor([-1], device=self.device)

        loss = self.criterior(positive_distance, negative_distance, target)
        return loss

    def validation_step(self, batch:Iterable[Tuple[int, int, int]], _):
        """Validation"""
        negative_triples = self.create_negative_samples(batch, self.entity2id)

        positive_distance = self.forward(batch)
        negative_distance = self.forward(negative_triples)
        target = torch.tensor([-1], device=self.device)

        loss = self.criterior(positive_distance, negative_distance, target)

        """Validation hit@k, mrr"""
        head, relation, tail = batch
        batch_size = len(head)

        entity_ids = torch.arange(end=len(self.entity2id), device=self.device).repeat(batch_size, 1)
        num_possible_triples = entity_ids.size()[1]

        # adjust shape
        heads = head.reshape(-1, 1).repeat(1, num_possible_triples)
        relations = relation.reshape(-1, 1).repeat(1, num_possible_triples)
        tails = tail.reshape(-1, 1).repeat(1, num_possible_triples)

        # modify tail entities
        tail_triples = torch.stack((heads, relations, entity_ids), dim=0).reshape(3, -1)
        tail_predictions = self.forward(tail_triples).reshape(len(heads), -1)

        # modify head entities
        head_triples = torch.stack((entity_ids, relations, tails), dim=0).reshape(3, -1)
        head_predictions = self.forward(head_triples).reshape(len(heads), -1)

        predictions = torch.cat((tail_predictions, head_predictions), dim=0)
        ground_truth_idxes = torch.cat((tail.reshape(-1, 1), head.reshape(-1, 1)))

        hit_at_1 = metrics.hit_at_k(predictions, ground_truth_idxes, device=self.device, k=1)
        hit_at_3 = metrics.hit_at_k(predictions, ground_truth_idxes, device=self.device, k=3)
        hit_at_5 = metrics.hit_at_k(predictions, ground_truth_idxes, device=self.device, k=5)
        hit_at_10 = metrics.hit_at_k(predictions, ground_truth_idxes, device=self.device, k=10)
        hit_at_20 = metrics.hit_at_k(predictions, ground_truth_idxes, device=self.device, k=20)
        mrr = metrics.mrr(predictions, ground_truth_idxes)

        result= dict(
            val_loss=loss,
            hit_at_1=hit_at_1,
            hit_at_3=hit_at_3,
            hit_at_5=hit_at_5,
            hit_at_10=hit_at_10,
            hit_at_20=hit_at_20,
            mrr=mrr,
        ) 
        self.log_dict(result)
        return result

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    ########################
    # Utility functions
    ########################

    def normalize_entity_embedding(self):
        self.entity_embedding.weight.data /= len(self.entity2id)

    def _distance(self, triples:Iterable[Tuple[int, int, int]]):
        head, relation, tail = triples
        
        # embedding layer
        head_embedding = self.entity_embedding(head)
        relation_embedding = self.relation_embedding(relation)
        tail_embedding = self.entity_embedding(tail)
        
        # calculate norm
        result = (head_embedding+relation_embedding) - tail_embedding
        norm = result.norm(p=self.norm, dim=1)
        return norm

    def create_negative_samples(self, triples:Iterable[Tuple[int, int, int]], entity2id:Dict[str, int]):
        size = (len(triples[0]),)

        # Negative sample을 만들 때 head를 바꿀지 tail을 바꿀지 선택
        target_entity_position = torch.randint(high=2, size=size, device=self.device)

        # 바꿀 random entity indices 생성
        random_entity_ids = torch.randint(low=0, high=len(entity2id), size=size, device=self.device)

        # Random entity로 교체
        heads, relations, tails =triples
        modified_heads = torch.where(target_entity_position==1, random_entity_ids, heads)
        modified_tails = torch.where(target_entity_position==0, random_entity_ids, tails)

        return [modified_heads, relations, modified_tails]


if __name__ == "__main__":
    from datasets import FB15k
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
    


