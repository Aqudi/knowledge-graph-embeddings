from argparse import ArgumentParser
import pytorch_lightning as pl
import torch

from datasets.FB15k import FB15k
from models.transE import TransE, TransEDataModule

# Argument parser
parser = ArgumentParser()

# Trainer specific args
parser.add_argument("--batch_size", type=int, default=None)

device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

pl.Trainer.add_argparse_args(parser)
TransE.add_model_specific_args(parser)

# Parse args
args = parser.parse_args()
print(args.__dict__)

# Dataset & Dataloader
entity2id, relation2id = {}, {}
train_dataset = FB15k(
    "./data/FB15k/freebase_mtr100_mte100-train.txt",
    entity2id=entity2id,
    relation2id=relation2id,
)
val_dataset = FB15k(
    "./data/FB15k/freebase_mtr100_mte100-valid.txt",
    entity2id=entity2id,
    relation2id=relation2id,
)
test_dataset = FB15k(
    "./data/FB15k/freebase_mtr100_mte100-test.txt",
    entity2id=entity2id,
    relation2id=relation2id,
)


datamodule = TransEDataModule(
    args.batch_size,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    val_dataset=val_dataset,
)

# Model
model = TransE(entity2id=entity2id, relation2id=relation2id, **vars(args))

# Trainer
trainer = pl.Trainer.from_argparse_args(args)

tune_result = trainer.tune(
    model,
    datamodule=datamodule,
)

trainer.fit(
    model,
    datamodule=datamodule,
)
