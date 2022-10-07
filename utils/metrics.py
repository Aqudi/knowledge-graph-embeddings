import torch


def hit_at_k(predictions, ground_truth_idxes, device, k=10):
    zero = torch.zeros(1, device=device)
    one = torch.ones(1, device=device)

    _, indices = predictions.topk(k=k, largest=False)
    return torch.where(indices == ground_truth_idxes, one, zero).sum().item()


def mrr(predictions, ground_truth_idxes):
    indices = predictions.argsort()
    return (
        (1.0 / (indices == ground_truth_idxes).nonzero()[:, 1].float().add(1.0))
        .sum()
        .item()
    )
