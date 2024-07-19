from __future__ import annotations

import abc
from typing import Self

import torch
import torch.nn.functional as F


def squared_distance(
    query_embed: torch.Tensor, candidate_embed: torch.Tensor
) -> torch.Tensor:
    return F.pairwise_distance(query_embed, candidate_embed) ** 2 / 2


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
    def __init__(self: Self, *, hard_negatives_ratio: int | None = None) -> None:
        super().__init__()
        self.hard_negatives_ratio = hard_negatives_ratio

    @abc.abstractmethod
    def forward(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ): ...

    @staticmethod
    def _check_embeds(user_embed: torch.Tensor, item_embed: torch.Tensor) -> None:
        assert user_embed.dim() == 2
        assert user_embed.dim() == 2
        assert user_embed.size(0) <= item_embed.size(0)
        assert user_embed.size(1) == item_embed.size(1)

    @staticmethod
    def _check_label(
        user_embed: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        if label is None:
            label = torch.ones(user_embed.size(0))
        else:
            assert label.dim() == 1
            assert label.size(0) == user_embed.size(0)
        return label

    @staticmethod
    def _check_sample_weight(
        user_embed: torch.Tensor, sample_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        if sample_weight is None:
            sample_weight = torch.ones(user_embed.size(0))
        else:
            assert sample_weight.dim() == 1
            assert sample_weight.size(0) == user_embed.size(0)
            assert (sample_weight >= 0).all()
        return sample_weight

    @staticmethod
    def _check_idx(
        size: int,
        idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if idx is None:
            idx = torch.arange(size)
        else:
            assert idx.dim() == 1
            assert idx.size(0) == size
        return idx

    @staticmethod
    def negative_masks(
        losses: torch.Tensor,
        *,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_idx = EmbeddingLoss._check_idx(losses.size(0), user_idx)
        item_idx = EmbeddingLoss._check_idx(losses.size(1), item_idx)
        # accidental hits can be samples with same user or item
        batch_size, num_items = losses.size()
        # pad columns with zeroes if num_items > batch_size
        user_hits = (
            user_idx[:, None] == F.pad(user_idx, (0, num_items - batch_size))[None, :]
        )
        # shape: (batch_size, num_items)
        # limit rows to batch size if num_items > batch_size
        item_hits = item_idx[:batch_size, None] == item_idx[None, :]
        # shape: (batch_size, num_items)
        accidental_hits = user_hits | item_hits
        # shape: (batch_size, num_items)
        return ~accidental_hits

    @staticmethod
    def hard_negative_mining(
        losses: torch.Tensor,
        negative_masks: torch.Tensor,
        *,
        hard_negatives_ratio: float | None = None,
    ) -> torch.Tensor:
        if hard_negatives_ratio is None:
            return losses, negative_masks

        # num_hard_negatives as a ratio of batch_size
        # important to handle different batch_size, especially last batch
        num_hard_negatives = int(losses.size(0) * hard_negatives_ratio)
        if hard_negatives_ratio > 1 and losses.size(1) <= num_hard_negatives:
            return losses, negative_masks

        # negative masks log will be 0 or -inf
        hard_negetives = (losses + negative_masks.log()).topk(
            k=num_hard_negatives, dim=1, sorted=False
        )
        losses = losses.gather(dim=1, index=hard_negetives.indices)
        # shape: (batch_size, num_hard_negatives)
        negative_masks = negative_masks.gather(dim=1, index=hard_negetives.indices)
        # shape: (batch_size, num_hard_negatives)
        return losses, negative_masks

    @staticmethod
    def alignment_loss(
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = user_embed.size(0)
        loss = squared_distance(user_embed, item_embed[:batch_size]) * label
        # shape: (batch_size)
        return weighted_mean(loss, sample_weight)

    @staticmethod
    def uniformity_loss(
        embed: torch.Tensor,
        *,
        idx: torch.Tensor | None = None,
        hard_negatives_ratio: float | None = None,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        sq_distances = squared_distance(embed[:, None, :], embed[None, :, :])
        # shape: (batch_size, num_items)
        losses = sq_distances * -sigma
        # shape: (batch_size, num_items)
        # take upper triangle
        negative_masks = EmbeddingLoss.negative_masks(losses, user_idx=idx).triu(
            diagonal=1
        )
        # shape: (batch_size, num_items)
        losses, negative_masks = EmbeddingLoss.hard_negative_mining(
            losses.reshape(1, -1),
            negative_masks.reshape(1, -1),
            hard_negatives_ratio=hard_negatives_ratio,
        )
        # shape: (1, num_hard_negatives | batch_size * num_items)
        denominator = negative_masks.sum() + 1e-10
        # shape: scalar
        return (losses + negative_masks.log() - denominator.log()).logsumexp(dim=1)

    @staticmethod
    def contrastive_loss(
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        sample_weight: torch.Tensor,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
        hard_negatives_ratio: float | None = None,
        sigma: float = 1.0,
        margin: float = 1.0,
    ) -> torch.Tensor:
        sq_distances = squared_distance(user_embed[:, None, :], item_embed[None, :, :])
        # shape: (batch_size, num_items)
        losses = (margin - sq_distances * sigma).relu()
        # shape: (batch_size, num_items)
        negative_masks = EmbeddingLoss.negative_masks(
            losses, user_idx=user_idx, item_idx=item_idx
        )
        # shape: (batch_size, num_items)
        losses, negative_masks = EmbeddingLoss.hard_negative_mining(
            losses, negative_masks, hard_negatives_ratio=hard_negatives_ratio
        )
        # shape: (batch_size, num_hard_negatives | num_items)
        loss = weighted_mean(losses, negative_masks, dim=1)
        # shape: (batch_size)
        return weighted_mean(loss, sample_weight)

    @staticmethod
    def infonce_loss(
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor,
        sample_weight: torch.Tensor,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
        hard_negatives_ratio: float | None = None,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        sq_distances = squared_distance(user_embed[:, None, :], item_embed[None, :, :])
        # shape: (batch_size, num_items)
        losses = sq_distances * label[:, None] * -sigma
        # shape: (batch_size, num_items)
        pos_loss = losses.diag()
        # shape: (batch_size)
        negative_masks = EmbeddingLoss.negative_masks(
            losses, user_idx=user_idx, item_idx=item_idx
        )
        # shape: (batch_size, num_items)
        losses, negative_masks = EmbeddingLoss.hard_negative_mining(
            losses, negative_masks, hard_negatives_ratio=hard_negatives_ratio
        )
        # shape: (batch_size, num_hard_negatives | num_items)
        logits = torch.cat([pos_loss[:, None], losses + negative_masks.log()], dim=1)
        # shape: (batch_size, num_hard_negatives | num_items + 1)
        loss = F.cross_entropy(
            logits, torch.zeros(logits.size(0), dtype=torch.long), reduction="none"
        )
        # shape: (batch_size)
        return weighted_mean(loss, sample_weight)

    @staticmethod
    def mine_loss(
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor,
        sample_weight: torch.Tensor,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
        hard_negatives_ratio: float | None = None,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        sq_distances = squared_distance(user_embed[:, None, :], item_embed[None, :, :])
        # shape: (batch_size, num_items)
        losses = sq_distances * label[:, None] * -sigma
        # shape: (batch_size, num_items)
        pos_loss = losses.diag()
        # shape: (batch_size)
        negative_masks = EmbeddingLoss.negative_masks(
            losses, user_idx=user_idx, item_idx=item_idx
        )
        # shape: (batch_size, num_items)
        losses, negative_masks = EmbeddingLoss.hard_negative_mining(
            losses, negative_masks, hard_negatives_ratio=hard_negatives_ratio
        )
        # shape: (batch_size, num_hard_negatives | num_items)
        negative_score = (losses + negative_masks.log()).logsumexp(dim=1)
        # shape: (batch_size)
        loss = -pos_loss + negative_score
        # shape: (batch_size)
        return weighted_mean(loss, sample_weight)


class AlignmentLoss(EmbeddingLoss):
    def forward(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_embeds(user_embed, item_embed)
        label = self._check_label(user_embed, label=label)
        sample_weight = self._check_sample_weight(
            user_embed, sample_weight=sample_weight
        )
        return self.alignment_loss(
            user_embed, item_embed, label=label, sample_weight=sample_weight
        )


class UniformityLoss(EmbeddingLoss):
    def forward(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_embeds(user_embed, item_embed)
        item_uniformity_loss = self.uniformity_loss(
            item_embed, idx=item_idx, hard_negatives_ratio=self.hard_negatives_ratio
        )
        user_uniformity_loss = self.uniformity_loss(
            user_embed, idx=user_idx, hard_negatives_ratio=self.hard_negatives_ratio
        )
        return (item_uniformity_loss + user_uniformity_loss) / 2


class AlignmentUniformityLoss(EmbeddingLoss):
    def forward(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_embeds(user_embed, item_embed)
        label = self._check_label(user_embed, label=label)
        sample_weight = self._check_sample_weight(
            user_embed, sample_weight=sample_weight
        )
        alingment_loss = self.alignment_loss(
            user_embed, item_embed, label=label, sample_weight=sample_weight
        )
        item_uniformity_loss = self.uniformity_loss(
            item_embed, idx=item_idx, hard_negatives_ratio=self.hard_negatives_ratio
        )
        user_uniformity_loss = self.uniformity_loss(
            user_embed, idx=user_idx, hard_negatives_ratio=self.hard_negatives_ratio
        )
        return alingment_loss + (item_uniformity_loss + user_uniformity_loss) / 2


class ContrastiveLoss(EmbeddingLoss):
    def forward(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_embeds(user_embed, item_embed)
        sample_weight = self._check_sample_weight(
            user_embed, sample_weight=sample_weight
        )

        return self.contrastive_loss(
            user_embed,
            item_embed,
            sample_weight=sample_weight,
            user_idx=user_idx,
            item_idx=item_idx,
            hard_negatives_ratio=self.hard_negatives_ratio,
        )


class AlignmentContrastiveLoss(EmbeddingLoss):
    def forward(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_embeds(user_embed, item_embed)
        label = self._check_label(user_embed, label=label)
        sample_weight = self._check_sample_weight(
            user_embed, sample_weight=sample_weight
        )

        alignment_loss = self.alignment_loss(
            user_embed, item_embed, label=label, sample_weight=sample_weight
        )
        contrastive_loss = self.contrastive_loss(
            user_embed,
            item_embed,
            sample_weight=sample_weight,
            user_idx=user_idx,
            item_idx=item_idx,
            hard_negatives_ratio=self.hard_negatives_ratio,
        )
        return alignment_loss + contrastive_loss


class InfomationNoiseContrastiveEstimationLoss(EmbeddingLoss):
    def forward(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_embeds(user_embed, item_embed)
        label = self._check_label(user_embed, label=label)
        sample_weight = self._check_sample_weight(
            user_embed, sample_weight=sample_weight
        )
        return self.infonce_loss(
            user_embed,
            item_embed,
            label=label,
            sample_weight=sample_weight,
            user_idx=user_idx,
            item_idx=item_idx,
            hard_negatives_ratio=self.hard_negatives_ratio,
        )


class MutualInformationNeuralEstimationLoss(EmbeddingLoss):
    def forward(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_embeds(user_embed, item_embed)
        label = self._check_label(user_embed, label=label)
        sample_weight = self._check_sample_weight(
            user_embed, sample_weight=sample_weight
        )
        return self.mine_loss(
            user_embed,
            item_embed,
            label=label,
            sample_weight=sample_weight,
            user_idx=user_idx,
            item_idx=item_idx,
            hard_negatives_ratio=self.hard_negatives_ratio,
        )


class PairwiseEmbeddingLoss(EmbeddingLoss, abc.ABC):
    def __init__(
        self: Self,
        *,
        hard_negatives_ratio: float | None = None,
        sigma: float = 1.0,
        margin: float = 1.0,
    ) -> None:
        super().__init__(hard_negatives_ratio=hard_negatives_ratio)
        self.sigma = sigma
        self.margin = margin

    @abc.abstractmethod
    def score_loss_fn(self: Self, score: torch.Tensor) -> torch.Tensor: ...

    def pariwise_loss(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor,
        sample_weight: torch.Tensor,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        hard_negatives_ratio: int | None = None,
        sigma: float = 1.0,
        margin: float = 1.0,
    ) -> torch.Tensor:
        sq_distances = (
            squared_distance(user_embed[:, None, :], item_embed[None, :, :])
            * label[:, None]
        )
        # shape: (batch_size, num_items)
        distances_diff = sq_distances - sq_distances.diag()[:, None]
        # shape: (batch_size, num_items)
        losses = self.score_loss_fn(distances_diff * sigma - margin)
        # shape: (batch_size, num_items)
        negative_masks = self.negative_masks(
            losses, user_idx=user_idx, item_idx=item_idx
        )
        # shape: (batch_size, num_items)
        losses, negative_masks = EmbeddingLoss.hard_negative_mining(
            losses, negative_masks, hard_negatives_ratio=hard_negatives_ratio
        )
        # shape: (batch_size, num_hard_negatives | num_items)
        loss = weighted_mean(losses, negative_masks, dim=1)
        # shape: (batch_size)
        return weighted_mean(loss, sample_weight)

    def forward(
        self: Self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_embeds(user_embed, item_embed)
        label = self._check_label(user_embed, label=label)
        sample_weight = self._check_sample_weight(
            user_embed, sample_weight=sample_weight
        )
        return self.pariwise_loss(
            user_embed,
            item_embed,
            label=label,
            sample_weight=sample_weight,
            user_idx=user_idx,
            item_idx=item_idx,
            hard_negatives_ratio=self.hard_negatives_ratio,
            sigma=self.sigma,
            margin=self.margin,
        )


class PairwiseLogisticLoss(PairwiseEmbeddingLoss):
    def score_loss_fn(self: Self, score: torch.Tensor) -> torch.Tensor:
        return -F.logsigmoid(score)


class PairwiseHingeLoss(PairwiseEmbeddingLoss):
    def score_loss_fn(self: Self, score: torch.Tensor) -> torch.Tensor:
        return (-score).relu()
