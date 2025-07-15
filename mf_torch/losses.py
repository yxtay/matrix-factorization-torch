from __future__ import annotations

import abc

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
        self,
        *,
        num_negatives: int | None = None,
        sigma: float = 1.0,
        margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_negatives = num_negatives
        self.sigma = sigma
        self.margin = margin

    def forward(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        target: torch.Tensor,
        *,
        item_idx: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        # target shape: (num_users, num_items)
        self.check_inputs(user_embed, item_embed, target)

        # row_idx, col_idx = target.indices()
        # # shape: (2, num_target)
        # col_idx = torch.cat([col_idx, torch.arange(item_embed.size(0)).to(col_idx)])
        # # shape: (num_target + num_items,)

        # user_embed = user_embed[row_idx, :]
        # # shape: (num_target, embed_size)
        # item_embed = item_embed[col_idx, :]
        # # shape: (num_target + num_items, embed_size)
        # target = target.values()
        # # shape: (num_target,)
        return self.loss(
            user_embed, item_embed, target, item_idx=item_idx, pos_idx=pos_idx
        )

    def check_inputs(
        self, user_embed: torch.Tensor, item_embed: torch.Tensor, target: torch.Tensor
    ) -> None:
        if user_embed.dim() != 2 or item_embed.dim() != 2:  # noqa: PLR2004
            msg = (
                "inputs should have 2 dimensions: "
                f"{user_embed.dim() = }, {item_embed.dim() = }"
            )
            raise ValueError(msg)

        if user_embed.size(1) != item_embed.size(1):
            msg = (
                "embeddings dimension 1 should match: "
                f"{ user_embed.size(1) = }, { item_embed.size(1) = }"
            )
            raise ValueError(msg)

        if not (
            user_embed.size(0) == target.size(0)
            and item_embed.size(0) >= target.size(0)
        ):
            msg = (
                "embeddings dimension 0 should match: "
                f"{target.size(0) = }, {user_embed.size(0) = }, {item_embed.size(0) = }"
            )
            raise ValueError(msg)

    @abc.abstractmethod
    def loss(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        target: torch.Tensor,
        *,
        item_idx: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor: ...

    def negative_masks(
        self,
        logits: torch.Tensor,
        *,
        item_idx: torch.Tensor,
        pos_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # accidental hits can be samples with same user or item
        batch_size = logits.size(0)
        # limit rows to batch size if num_items > batch_size
        accidental_hits = item_idx[:batch_size, None] == item_idx[None, :]
        # shape: (batch_size, num_items)
        if pos_idx is not None:
            # shape: (batch_size, num_positives)
            # mask shape: (batch_size, num_items, num_positives)
            accidental_hits |= (pos_idx.unsqueeze(1) == item_idx[None, :, None]).any(-1)
            # shape: (batch_size, num_items)
        return ~accidental_hits

    @torch.no_grad()
    def hard_mining(
        self, logits: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        if self.num_negatives is None:
            return negative_masks

        if self.num_negatives >= logits.size(1):
            return negative_masks

        # negative masks log will be 0 or -inf
        indices = (
            (logits + negative_masks.log())
            .topk(k=self.num_negatives, dim=-1, sorted=False)
            .indices
        )
        return torch.scatter(torch.zeros_like(negative_masks), -1, indices, 1.0)

    @torch.no_grad()
    def semi_hard_mining(
        self, logits: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        if self.num_negatives is None:
            return negative_masks

        if self.num_negatives >= logits.size(1):
            return negative_masks

        logits_mod = logits - logits.diag()[:, None]
        # shape: (batch_size, num_items)
        # neg: semi hard negatives, descending order first, so minus minimum value
        # pos: hard negatives, ascending order later, so take negative value
        logits_min = logits_mod.min(dim=-1, keepdim=True)[0]
        logits_mod = torch.where(logits_mod < 0, logits_mod - logits_min, -logits_mod)
        # shape: (batch_size, num_items)
        # negative masks log will be 0 or -inf, so false negatives will be last
        logits_mod = logits_mod + negative_masks.log()
        # shape: (batch_size, num_items)

        # pos: semi hard negatives, neg: hard negatives, -inf: false negatives
        # so take largest
        indices = logits_mod.topk(k=self.num_negatives, dim=-1, sorted=False).indices
        # shape: (batch_size, num_items)
        negative_masks &= torch.scatter(
            torch.zeros_like(negative_masks), -1, indices, 1.0
        )
        return negative_masks

    def alignment_loss(
        self, user_embed: torch.Tensor, item_embed: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        batch_size = user_embed.size(0)
        loss = squared_distance(user_embed, item_embed[:batch_size]).diag()
        # shape: (batch_size)
        return (loss * target * self.sigma).sum()

    def contrastive_loss(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        target: torch.Tensor,
        *,
        item_idx: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        logits = -squared_distance(user_embed, item_embed)
        # shape: (batch_size, num_items)
        logits = logits * target.sign()[:, None] * self.sigma
        # shape: (batch_size, num_items)
        negative_masks = self.negative_masks(logits, item_idx=item_idx, pos_idx=pos_idx)
        # shape: (batch_size, num_items)
        negative_masks = self.semi_hard_mining(logits, negative_masks)
        # shape: (batch_size, num_items)
        losses = (logits + target.sign()[:, None] * self.margin).relu()
        # shape: (batch_size, num_items)
        loss = weighted_mean(losses, negative_masks, dim=-1)
        # shape: (batch_size)
        return (loss * target.abs()).sum()

    def infonce_loss(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        target: torch.Tensor,
        *,
        item_idx: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        logits = -squared_distance(user_embed, item_embed)
        # shape: (batch_size, num_items)
        logits = logits * target.sign()[:, None] * self.sigma
        # shape: (batch_size, num_items)
        negative_masks = self.negative_masks(logits, item_idx=item_idx, pos_idx=pos_idx)
        # shape: (batch_size, num_items)
        negative_masks = self.semi_hard_mining(logits, negative_masks)
        # shape: (batch_size, num_items)
        pos_logit = logits.diag()[:, None]
        # shape: (batch_size)
        logits = torch.cat([pos_logit, logits + negative_masks.log()], dim=-1)
        # shape: (batch_size, num_items + 1)
        loss = F.cross_entropy(
            logits,
            torch.zeros(logits.size(0), dtype=torch.long, device=logits.device),
            reduction="none",
        )
        # shape: (batch_size)
        return (loss * target.abs()).sum()

    def mine_loss(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        target: torch.Tensor,
        *,
        item_idx: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        logits = -squared_distance(user_embed, item_embed)
        # shape: (batch_size, num_items)
        logits = logits * target.sign()[:, None] * self.sigma
        # shape: (batch_size, num_items)
        negative_masks = self.negative_masks(logits, item_idx=item_idx, pos_idx=pos_idx)
        # shape: (batch_size, num_items)
        negative_masks = self.semi_hard_mining(logits, negative_masks)
        # shape: (batch_size, num_items)
        negative_score = (logits + negative_masks.log()).logsumexp(dim=-1)
        # shape: (batch_size)
        loss = -logits.diag() + negative_score
        # shape: (batch_size)
        return (loss * target.abs()).sum()


class AlignmentLoss(EmbeddingLoss):
    def loss(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        target: torch.Tensor,
        *,
        item_idx: torch.Tensor,  # noqa: ARG002
        pos_idx: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        return self.alignment_loss(user_embed, item_embed, target)


class ContrastiveLoss(EmbeddingLoss):
    def loss(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        target: torch.Tensor,
        *,
        item_idx: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.contrastive_loss(
            user_embed, item_embed, target, item_idx=item_idx, pos_idx=pos_idx
        )


class AlignmentContrastiveLoss(EmbeddingLoss):
    def loss(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        target: torch.Tensor,
        *,
        item_idx: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        alignment_loss = self.alignment_loss(user_embed, item_embed, target)
        contrastive_loss = self.contrastive_loss(
            user_embed, item_embed, target, item_idx=item_idx, pos_idx=pos_idx
        )
        return alignment_loss + contrastive_loss


class InfomationNoiseContrastiveEstimationLoss(EmbeddingLoss):
    def loss(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        target: torch.Tensor,
        *,
        item_idx: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.infonce_loss(
            user_embed, item_embed, target, item_idx=item_idx, pos_idx=pos_idx
        )


class MutualInformationNeuralEstimationLoss(EmbeddingLoss):
    def loss(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        target: torch.Tensor,
        *,
        item_idx: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.mine_loss(
            user_embed, item_embed, target, item_idx=item_idx, pos_idx=pos_idx
        )


class PairwiseEmbeddingLoss(EmbeddingLoss, abc.ABC):
    def loss(
        self,
        user_embed: torch.Tensor,
        item_embed: torch.Tensor,
        target: torch.Tensor,
        *,
        item_idx: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        logits = -squared_distance(user_embed, item_embed)
        # shape: (batch_size, num_items)
        logits = logits * target.sign()[:, None] * self.sigma
        # shape: (batch_size, num_items)
        negative_masks = self.negative_masks(logits, item_idx=item_idx, pos_idx=pos_idx)
        # shape: (batch_size, num_items)
        negative_masks = self.semi_hard_mining(logits, negative_masks)
        # shape: (batch_size, num_items)
        pos_logit = logits.diag()[:, None]
        # shape: (batch_size, 1)
        losses = self.score_loss_fn(logits - pos_logit + self.margin)
        # shape: (batch_size, num_items)
        loss = weighted_mean(losses, negative_masks, dim=-1)
        # shape: (batch_size)
        return (loss * target.abs()).sum()

    @abc.abstractmethod
    def score_loss_fn(self, score: torch.Tensor) -> torch.Tensor: ...


class PairwiseLogisticLoss(PairwiseEmbeddingLoss):
    def score_loss_fn(self, score: torch.Tensor) -> torch.Tensor:
        return -F.logsigmoid(-score)


class PairwiseHingeLoss(PairwiseEmbeddingLoss):
    def score_loss_fn(self, score: torch.Tensor) -> torch.Tensor:
        return (score).relu()
