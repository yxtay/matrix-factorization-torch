from __future__ import annotations

import abc
from typing import Self

import torch
import torch.nn.functional as F  # noqa: N812


def squared_distance(
    query_embed: torch.Tensor, candidate_embed: torch.Tensor
) -> torch.Tensor:
    return torch.cdist(query_embed, candidate_embed) ** 2 / 2


def weighted_mean(
    values: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    denominator = sample_weights.sum(dim=dim, keepdim=True) + 1e-10
    return (values * sample_weights / denominator).sum(dim=dim, keepdim=keepdim)


class EmbeddingLoss(torch.nn.Module, abc.ABC):
    def __init__(
        self: Self,
        *,
        hard_negatives_ratio: int | None = None,
        sigma: float = 1.0,
        margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.hard_negatives_ratio = hard_negatives_ratio
        self.sigma = sigma
        self.margin = margin

    def forward(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        self.check_inputs(user_embed, item_embed, targets)

        indices = targets.indices()
        row_idx = indices[0, :]
        col_idx = torch.cat(
            [indices[1, :], torch.arange(item_embed.size(0)).to(indices)]
        )

        user_embed = user_embed[row_idx, :]
        item_embed = item_embed[col_idx, :]
        targets = targets.values()
        return self.loss(
            user_embed, item_embed, targets, user_idx=row_idx, item_idx=col_idx
        )

    def check_inputs(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if user_embed.dim() != 2 or item_embed.dim() != 2 or targets.dim() != 2:  # noqa: PLR2004
            msg = f"inputs should have 2 dimensions: {user_embed.dim() = }, {item_embed.dim() = }, {targets.dim() = }"
            raise ValueError(msg)

        if user_embed.size(1) != item_embed.size(1):
            msg = f"embedding dimensions should match: { user_embed.size(1) = }, { item_embed.size(1) = }"
            raise ValueError(msg)

        if (user_embed.size(0), item_embed.size(0)) != targets.size():
            msg = f"embedding dimension 1 should match targets dimensions: {(user_embed.size(0), item_embed.size(0)) = }, {targets.size() = }"
            raise ValueError(msg)

    @abc.abstractmethod
    def loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
        *,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor: ...

    def negative_masks(
        self: Self,
        losses: torch.Tensor,
        *,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # accidental hits can be samples with same user or item
        batch_size, num_items = losses.size()
        # pad columns with zeroes if num_items > batch_size
        accidental_hits = (
            user_idx[:, None] == F.pad(user_idx, (0, num_items - batch_size))[None, :]
        )
        # shape: (batch_size, num_items)
        if item_idx is not None:
            # limit rows to batch size if num_items > batch_size
            accidental_hits |= item_idx[:batch_size, None] == item_idx[None, :]
            # shape: (batch_size, num_items)
        return ~accidental_hits

    def hard_negative_mining(
        self: Self, losses: torch.Tensor, negative_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hard_negatives_ratio is None:
            return losses, negative_masks

        # num_hard_negatives as a ratio of batch_size
        # important to handle different batch_size, especially last batch
        num_hard_negatives = int(losses.size(0) * self.hard_negatives_ratio)
        if self.hard_negatives_ratio > 1 and losses.size(1) <= num_hard_negatives:
            return losses, negative_masks

        # negative masks log will be 0 or -inf
        hard_negetives = (losses + negative_masks.log()).topk(
            k=num_hard_negatives, dim=-1, sorted=False
        )
        losses = losses.gather(dim=-1, index=hard_negetives.indices)
        # shape: (batch_size, num_hard_negatives)
        negative_masks = negative_masks.gather(dim=-1, index=hard_negetives.indices)
        # shape: (batch_size, num_hard_negatives)
        return losses, negative_masks

    def alignment_loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = user_embed.size(0)
        loss = (
            squared_distance(user_embed, item_embed[:batch_size]).diag()
            * targets.sign()
        )
        # shape: (batch_size)
        return weighted_mean(loss, targets.abs())

    def uniformity_loss(
        self: Self, embed: torch.Tensor, *, idx: torch.Tensor
    ) -> torch.Tensor:
        logits = -squared_distance(embed, embed)
        # shape: (batch_size, num_items)
        losses = logits * self.sigma
        # shape: (batch_size, num_items)
        # take upper triangle
        negative_masks = self.negative_masks(losses, user_idx=idx).triu(diagonal=1)
        # shape: (batch_size, num_items)
        losses, negative_masks = self.hard_negative_mining(
            losses.reshape(1, -1), negative_masks.reshape(1, -1)
        )
        # shape: (1, num_hard_negatives | batch_size * num_items)
        denominator = negative_masks.sum() + 1e-10
        # shape: scalar
        return (losses + negative_masks.log() - denominator.log()).logsumexp(dim=-1)

    def contrastive_loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
        *,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        logits = -squared_distance(user_embed, item_embed)
        # shape: (batch_size, num_items)
        losses = (self.margin - logits * self.sigma).relu()
        # shape: (batch_size, num_items)
        negative_masks = self.negative_masks(
            losses, user_idx=user_idx, item_idx=item_idx
        )
        # shape: (batch_size, num_items)
        losses, negative_masks = self.hard_negative_mining(losses, negative_masks)
        # shape: (batch_size, num_hard_negatives | num_items)
        loss = weighted_mean(losses, negative_masks, dim=-1)
        # shape: (batch_size)
        return weighted_mean(loss, targets.abs())

    def infonce_loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
        *,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        logits = -squared_distance(user_embed, item_embed)
        # shape: (batch_size, num_items)
        losses = logits * targets.sign()[:, None] * self.sigma
        # shape: (batch_size, num_items)
        pos_loss = losses.diag()
        # shape: (batch_size)
        negative_masks = self.negative_masks(
            losses, user_idx=user_idx, item_idx=item_idx
        )
        # shape: (batch_size, num_items)
        losses, negative_masks = self.hard_negative_mining(losses, negative_masks)
        # shape: (batch_size, num_hard_negatives | num_items)
        logits = torch.cat([pos_loss[:, None], losses + negative_masks.log()], dim=-1)
        # shape: (batch_size, num_hard_negatives | num_items + 1)
        loss = F.cross_entropy(
            logits, torch.zeros(logits.size(0), dtype=torch.long), reduction="none"
        )
        # shape: (batch_size)
        return weighted_mean(loss, targets.abs())

    def mine_loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
        *,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        logits = -squared_distance(user_embed, item_embed)
        # shape: (batch_size, num_items)
        losses = logits * targets.sign()[:, None] * self.sigma
        # shape: (batch_size, num_items)
        pos_loss = losses.diag()
        # shape: (batch_size)
        negative_masks = self.negative_masks(
            losses, user_idx=user_idx, item_idx=item_idx
        )
        # shape: (batch_size, num_items)
        losses, negative_masks = self.hard_negative_mining(losses, negative_masks)
        # shape: (batch_size, num_hard_negatives | num_items)
        negative_score = (losses + negative_masks.log()).logsumexp(dim=-1)
        # shape: (batch_size)
        loss = -pos_loss + negative_score
        # shape: (batch_size)
        return weighted_mean(loss, targets.abs())


class AlignmentLoss(EmbeddingLoss):
    def loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
        *,
        user_idx: torch.Tensor,  # noqa: ARG002
        item_idx: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        return self.alignment_loss(user_embed, item_embed, targets)


class UniformityLoss(EmbeddingLoss):
    def loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,  # noqa: ARG002
        *,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        item_uniformity_loss = self.uniformity_loss(item_embed, idx=item_idx)
        user_uniformity_loss = self.uniformity_loss(user_embed, idx=user_idx)
        return (item_uniformity_loss + user_uniformity_loss) / 2


class AlignmentUniformityLoss(EmbeddingLoss):
    def loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
        *,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        alingment_loss = self.alignment_loss(user_embed, item_embed, targets)
        item_uniformity_loss = self.uniformity_loss(item_embed, idx=item_idx)
        user_uniformity_loss = self.uniformity_loss(user_embed, idx=user_idx)
        return alingment_loss + (item_uniformity_loss + user_uniformity_loss) / 2


class ContrastiveLoss(EmbeddingLoss):
    def loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
        *,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.contrastive_loss(
            user_embed, item_embed, targets, user_idx=user_idx, item_idx=item_idx
        )


class AlignmentContrastiveLoss(EmbeddingLoss):
    def loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
        *,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        alignment_loss = self.alignment_loss(user_embed, item_embed, targets)
        contrastive_loss = self.contrastive_loss(
            user_embed, item_embed, targets, user_idx=user_idx, item_idx=item_idx
        )
        return alignment_loss + contrastive_loss


class InfomationNoiseContrastiveEstimationLoss(EmbeddingLoss):
    def loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
        *,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.infonce_loss(
            user_embed, item_embed, targets, user_idx=user_idx, item_idx=item_idx
        )


class MutualInformationNeuralEstimationLoss(EmbeddingLoss):
    def loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
        *,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.mine_loss(
            user_embed, item_embed, targets, user_idx=user_idx, item_idx=item_idx
        )


class PairwiseEmbeddingLoss(EmbeddingLoss, abc.ABC):
    def loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        targets: torch.Tensor,
        *,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        logits = -squared_distance(user_embed, item_embed) * targets.sign()[:, None]
        # shape: (batch_size, num_items)
        logits_diff = logits.diag()[:, None] - logits
        # shape: (batch_size, num_items)
        losses = self.score_loss_fn(logits_diff * self.sigma - self.margin)
        # shape: (batch_size, num_items)
        negative_masks = self.negative_masks(
            losses, user_idx=user_idx, item_idx=item_idx
        )
        # shape: (batch_size, num_items)
        losses, negative_masks = self.hard_negative_mining(losses, negative_masks)
        # shape: (batch_size, num_hard_negatives | num_items)
        loss = weighted_mean(losses, negative_masks, dim=-1)
        # shape: (batch_size)
        return weighted_mean(loss, targets.abs())

    @abc.abstractmethod
    def score_loss_fn(self: Self, score: torch.Tensor) -> torch.Tensor: ...


class PairwiseLogisticLoss(PairwiseEmbeddingLoss):
    def score_loss_fn(self: Self, score: torch.Tensor) -> torch.Tensor:
        return -F.logsigmoid(score)


class PairwiseHingeLoss(PairwiseEmbeddingLoss):
    def score_loss_fn(self: Self, score: torch.Tensor) -> torch.Tensor:
        return (-score).relu()
