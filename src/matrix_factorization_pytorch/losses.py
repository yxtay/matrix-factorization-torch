import abc

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
    @abc.abstractmethod
    def forward(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ):
        ...

    @staticmethod
    def _check_embeds(user_embed: torch.Tensor, item_embed: torch.Tensor) -> None:
        assert user_embed.dim() == 2
        assert user_embed.size() == item_embed.size()

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
        sample_weight: torch.Tensor, idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        if idx is None:
            idx = torch.arange(sample_weight.size(0))
        else:
            assert idx.dim() == 1
            assert idx.size(0) == sample_weight.size(0)
        return idx

    @staticmethod
    def negative_weights(
        sample_weight: torch.Tensor,
        *,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_idx = EmbeddingLoss._check_idx(sample_weight, user_idx)
        item_idx = EmbeddingLoss._check_idx(sample_weight, item_idx)

        # accidental hits can be samples with same user or item
        accidental_hits = (user_idx[None, :] == user_idx[:, None]) & (
            item_idx[None, :] == item_idx[:, None]
        )
        # shape: (batch_size, batch_size)
        return ~accidental_hits * sample_weight[None, :]

    @staticmethod
    def alignment_loss(
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        loss = squared_distance(user_embed, item_embed) * label
        # shape: (batch_size)
        return weighted_mean(loss, sample_weight)

    @staticmethod
    def uniformity_loss(
        embed: torch.Tensor,
        sample_weight: torch.Tensor,
        *,
        idx: torch.Tensor | None = None,
        sigma: float = 1.0,
        margin: float = 0.0,
    ) -> torch.Tensor:
        sq_distances = squared_distance(embed[None, :, :], embed[:, None, :])
        # shape: (batch_size, batch_size)
        losses = margin - sq_distances * sigma
        # shape: (batch_size, batch_size)
        # take upper triangle
        negative_weights = EmbeddingLoss.negative_weights(
            sample_weight, user_idx=idx
        ).triu(diagonal=1)
        # shape: (batch_size, batch_size)
        denominator = negative_weights.sum() + 1e-10
        # shape: scalar
        return (losses + negative_weights.log() - denominator.log()).logsumexp(
            dim=(0, 1)
        )

    @staticmethod
    def contrastive_loss(
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        sample_weight: torch.Tensor,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
        sigma: float = 1.0,
        margin: float = 1.0,
    ) -> torch.Tensor:
        sq_distances = squared_distance(user_embed[None, :, :], item_embed[:, None, :])
        # shape: (batch_size, batch_size)
        losses = (margin - sq_distances * sigma).relu()
        # shape: (batch_size, batch_size)

        # weighted mean over negative samples
        negative_weights = EmbeddingLoss.negative_weights(
            sample_weight, user_idx=user_idx, item_idx=item_idx
        )
        # shape: (batch_size, batch_size)
        loss = weighted_mean(losses, negative_weights, dim=-1)
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
        sigma: float = 1.0,
    ) -> torch.Tensor:
        sq_distances = (
            squared_distance(user_embed[None, :, :], item_embed[:, None, :])
            * label[:, None]
        )
        # shape: (batch_size, batch_size)
        negative_weights = EmbeddingLoss.negative_weights(
            sample_weight, user_idx=user_idx, item_idx=item_idx
        )
        # shape: (batch_size, batch_size)
        negative_score = (sq_distances * -sigma + negative_weights.log()).logsumexp(
            dim=-1
        )
        # shape: (batch_size)
        loss = sq_distances.diag() + negative_score
        # shape: (batch_size)
        return weighted_mean(loss, sample_weight)


class AlignmentLoss(EmbeddingLoss):
    def forward(
        self,
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


class ItemUniformityLoss(EmbeddingLoss):
    def forward(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sample_weight = self._check_sample_weight(
            item_embed, sample_weight=sample_weight
        )
        return self.uniformity_loss(item_embed, sample_weight, idx=item_idx)


class UserUniformityLoss(EmbeddingLoss):
    def forward(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        user_idx: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sample_weight = self._check_sample_weight(
            user_embed, sample_weight=sample_weight
        )
        return self.uniformity_loss(user_embed, sample_weight, idx=user_idx)


class AlignmentUniformityLoss(EmbeddingLoss):
    def forward(
        self,
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
            item_embed, sample_weight, idx=item_idx
        )
        user_uniformity_loss = self.uniformity_loss(
            user_embed, sample_weight, idx=user_idx
        )
        return alingment_loss + (item_uniformity_loss + user_uniformity_loss) / 2


class ContrastiveLoss(EmbeddingLoss):
    def forward(
        self,
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
        )


class AlignmentContrastiveLoss(EmbeddingLoss):
    def forward(
        self,
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
        )
        return alignment_loss + contrastive_loss


class MutualInformationNeuralEstimatorLoss(EmbeddingLoss):
    def forward(
        self,
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
        )


class PairwiseEmbeddingLoss(EmbeddingLoss, abc.ABC):
    def __init__(self, *, sigma: float = 1.0, margin: float = 1.0) -> None:
        super().__init__()
        self.sigma = sigma
        self.margin = margin

    @abc.abstractmethod
    def score_loss_fn(self, score) -> torch.Tensor:
        ...

    def pariwise_loss(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        *,
        label: torch.Tensor,
        sample_weight: torch.Tensor,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        sigma: float = 1.0,
        margin: float = 1.0,
    ) -> torch.Tensor:
        sq_distances = (
            squared_distance(user_embed[None, :, :], item_embed[:, None, :])
            * label[:, None]
        )
        # shape: (batch_size, batch_size)
        distances_diff = sq_distances - sq_distances.diag()[:, None]
        # shape: (batch_size, batch_size)
        losses = self.score_loss_fn(distances_diff * sigma - margin)
        # shape: (batch_size, batch_size)

        # weighted mean over negative samples
        negative_weights = self.negative_weights(
            sample_weight, user_idx=user_idx, item_idx=item_idx
        )
        # shape: (batch_size)
        loss = weighted_mean(losses, negative_weights, dim=-1)
        # shape: (batch_size)
        return weighted_mean(loss, sample_weight)

    def forward(
        self,
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
            sigma=self.sigma,
            margin=self.margin,
        )


class PairwiseLogisticLoss(PairwiseEmbeddingLoss):
    def score_loss_fn(self, score: torch.Tensor) -> torch.Tensor:
        return -F.logsigmoid(score)


class PairwiseHingeLoss(PairwiseEmbeddingLoss):
    def score_loss_fn(self, score: torch.Tensor) -> torch.Tensor:
        return (-score).relu()
