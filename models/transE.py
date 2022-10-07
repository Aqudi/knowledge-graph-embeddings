from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import pytorch_lightning as pl

from utils import metrics


class TransEDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size, train_dataset=None, test_dataset=None, val_dataset=None
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["train_dataset", "test_dataset", "val_dataset"]
        )

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset

    def collate_fn(self, batch):
        heads, relations, tails = [], [], []
        for h, r, t in batch:
            heads.append(h)
            relations.append(r)
            tails.append(t)
        return (
            torch.LongTensor(heads),
            torch.LongTensor(relations),
            torch.LongTensor(tails),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, self.hparams.batch_size, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, self.hparams.batch_size, collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, self.hparams.batch_size, collate_fn=self.collate_fn
        )


class TransE(pl.LightningModule):
    def __init__(
        self,
        embedding_dim=20,
        margin=1,
        lr=1e-5,
        norm=1,
        entity2id={},
        relation2id={},
        *args,
        **kwargs
    ):
        super().__init__()

        # parameters
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.lr = lr
        self.norm = norm
        self.entity2id = entity2id
        self.relation2id = relation2id

        # training setup
        self.criterior = nn.MarginRankingLoss(margin=margin)

        # layers
        self.relation_embedding = nn.Embedding(
            num_embeddings=len(self.relation2id),
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.entity_embedding = nn.Embedding(
            num_embeddings=len(self.entity2id),
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # initialize embeddings
        initializer_factor = np.divide(6, np.sqrt(embedding_dim))
        self.entity_embedding.weight.data.uniform_(
            -initializer_factor, initializer_factor
        )
        self.relation_embedding.weight.data.uniform_(
            -initializer_factor, initializer_factor
        )

        # normalize rel_embedding
        self.relation_embedding.weight.data = F.normalize(
            self.relation_embedding.weight.data
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("TransE")
        parser.add_argument("--embedding_dim", type=int, default=20)
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--margin", type=float, default=1)
        parser.add_argument("--norm", type=int, default=1)

    def forward(self, x):
        """Inference only"""
        return self._distance(x)

    def _step(self, batch, _):
        negative_triples = self.create_negative_samples(batch, self.entity2id)

        positive_distance = self.forward(batch)
        negative_distance = self.forward(negative_triples)

        target = torch.tensor([1], dtype=torch.long, device=self.device)

        return positive_distance, negative_distance, target

    def training_step(self, batch, _):
        """Training"""
        self.normalize_entity_embedding()
        positive_distance, negative_distance, target = self._step(batch, _)
        loss = self.criterior(positive_distance, negative_distance, target)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, _):
        """Validation"""
        positive_distance, negative_distance, target = self._step(batch, _)
        loss = self.criterior(positive_distance, negative_distance, target)

        """Validation hit@k, mrr"""
        head, relation, tail = batch
        batch_size = len(head)

        entity_ids = torch.arange(
            end=len(self.entity2id),
            device=self.device,
        ).repeat(batch_size, 1)
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

        ks = [1, 3, 5, 10, 20]
        hit_at_k = [
            metrics.hit_at_k(predictions, ground_truth_idxes, device=self.device, k=k)
            for k in ks
        ]

        mrr = metrics.mrr(predictions, ground_truth_idxes)

        result = dict(
            val_loss=loss,
            hit_at_1=hit_at_k[0],
            hit_at_3=hit_at_k[1],
            hit_at_5=hit_at_k[2],
            hit_at_10=hit_at_k[3],
            hit_at_20=hit_at_k[4],
            mrr=mrr,
        )
        self.log_dict(result, prog_bar=True)
        return result

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    ########################
    # Utility functions
    ########################

    def normalize_entity_embedding(self):
        self.entity_embedding.weight.data = F.normalize(
            self.entity_embedding.weight.data
        )

    def _distance(self, triples):
        head, relation, tail = triples

        # embedding layer
        head_embedding = self.entity_embedding(head)
        relation_embedding = self.relation_embedding(relation)
        tail_embedding = self.entity_embedding(tail)

        # calculate norm
        result = (head_embedding + relation_embedding) - tail_embedding
        norm = result.norm(p=self.norm, dim=1)
        return norm

    def create_negative_samples(self, triples, entity2id):
        heads, relations, tails = triples
        size = (1, len(heads))

        # Negative sample을 만들 때 head를 바꿀지 tail을 바꿀지 선택
        target_entity_position = torch.randint(high=2, size=size, device=self.device)
        target_entity_position = target_entity_position.squeeze()

        # 바꿀 random entity indices 생성
        random_entity_ids = torch.randint(
            low=0, high=len(entity2id), size=size, device=self.device
        )
        random_entity_ids = random_entity_ids.squeeze()

        # Random entity로 교체
        modified_heads = torch.where(
            target_entity_position == 1, random_entity_ids, heads
        )
        modified_tails = torch.where(
            target_entity_position == 0, random_entity_ids, tails
        )

        return (
            modified_heads,
            relations,
            modified_tails,
        )


if __name__ == "__main__":
    from datasets import FB15k
    from models.transE import TransE
    from torch.utils.data import DataLoader

    # Dataset & Dataloader
    entity2id, relation2id = {}, {}
    train_dataset = FB15k(
        "./data/FB15k/freebase_mtr100_mte100-train.txt",
        entity2id=entity2id,
        relation2id=relation2id,
    )
    test_dataset = FB15k(
        "./data/FB15k/freebase_mtr100_mte100-test.txt",
        entity2id=entity2id,
        relation2id=relation2id,
    )
    val_dataset = FB15k(
        "./data/FB15k/freebase_mtr100_mte100-valid.txt",
        entity2id=entity2id,
        relation2id=relation2id,
    )

    datamodule = TransEDataModule(
        10,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
    )

    # Model
    model = TransE(
        embedding_dim=10,
        margin=5,
        entity2id=entity2id,
        relation2id=relation2id,
    )

    positive_triples = next(iter(datamodule.train_dataloader()))
    negative_triples = model.create_negative_samples(positive_triples, entity2id)

    loss = model(positive_triples)
    print(loss)
