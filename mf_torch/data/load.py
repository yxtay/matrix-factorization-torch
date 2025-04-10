from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.data as torch_data
import torch.utils.data._utils.collate as torch_collate

from mf_torch.params import NUM_EMBEDDINGS, NUM_HASHES, PADDING_IDX

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Self

    import pyarrow.dataset as ds


def collate_features(
    value: None | str | int | dict[str, Any] | list[Any],
    key: str = "",
    feature_names: dict[str, str] | None = None,
) -> tuple[list[str], torch.Tensor]:
    if feature_names is None:
        feature_names = {}
    if value is None:
        # ignore null values
        return [], torch.zeros(0)

    if isinstance(value, (str | int)):
        # category feature
        return [f"{key}:{value}".lower()], torch.ones(1)

    if isinstance(value, list):
        # list feature, potentially nested
        values, weights = zip(
            *[collate_features(v, key, feature_names) for v in value], strict=True
        )
    elif isinstance(value, dict):
        # nested feature
        key_fmt = f"{key}:{{}}" if key else "{}"
        values, weights = zip(
            *[
                collate_features(v, key_fmt.format(feature_names[k]), feature_names)
                for k, v in value.items()
            ],
            strict=True,
        )

    num_values = sum(len(val) > 0 for val in values)
    values = list(itertools.chain(*values))
    weights = torch.concatenate(weights) / num_values
    return values, weights


def hash_features(
    values: list[str],
    weights: torch.Tensor,
    *,
    num_hashes: int = NUM_HASHES,
    num_embeddings: int = NUM_EMBEDDINGS,
) -> tuple[torch.Tensor, torch.Tensor]:
    import xxhash

    hashes = [
        xxhash.xxh32_intdigest(val, seed)
        for val in values
        for seed in range(num_hashes)
    ]
    hashes = torch.as_tensor(hashes) % (num_embeddings - 1) + 1
    weights = weights.repeat_interleave(num_hashes) / num_hashes
    return hashes.to(torch.int), weights


def select_fields(example: dict[str, Any], *, fields: list[str]) -> dict[str, Any]:
    return {key: example[key] for key in fields}


def nest_example(example: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return {key: example}


def merge_examples(examples: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for example in examples:
        merged.update(example)
    return merged


def embed_example(example: dict[str, Any], *, model: torch.nn.Module) -> dict[str, Any]:
    return {
        **example,
        "embedding": model(example["feature_hashes"], example["feature_weights"]).numpy(
            force=True
        ),
    }


def pad_jagged_tensors(
    tensors: list[torch.Tensor], pad_value: int = PADDING_IDX
) -> torch.Tensor:
    return torch.nested.nested_tensor(tensors).to_padded_tensor(padding=pad_value)


def pad_tensors(
    batch: Iterable[torch.Tensor],
    dim: int = -1,
    *,
    pad_left: bool = True,
    pad_value: int = PADDING_IDX,
) -> torch.Tensor:
    elem = next(iter(batch))
    pad_size = max(example.size(dim) for example in iter(batch))
    pad = [0] * (elem.dim() * 2)
    pad_dim = 2 * dim + 0 if pad_left else 1

    def pad_tensor(tensor: torch.Tensor) -> torch.Tensor:
        pad[pad_dim] = pad_size - tensor.size(dim)
        return F.pad(tensor, pad, value=pad_value)

    return torch.stack([pad_tensor(tensor) for tensor in batch])


def collate_tensor_fn(
    batch: Iterable[torch.Tensor], *, collate_fn_map: dict | None = None
) -> torch.Tensor:
    it = iter(batch)
    elem_size = next(it).size()
    if any(elem.size() != elem_size for elem in it):
        if all(elem.size()[:-1] == elem_size[:-1] for elem in it):
            # only last dimension different
            return pad_tensors(batch, dim=-1)
        if all(elem.size()[1:] == elem_size[1:] for elem in it):
            # only first dimension different
            return pad_tensors(batch, dim=0)

    return torch_collate.collate_tensor_fn(batch, collate_fn_map=collate_fn_map)


torch_collate.default_collate_fn_map[torch.Tensor] = collate_tensor_fn


@torch_data.functional_datapipe("load_parquet_as_dict")
class ParquetDictLoaderIterDataPipe(torch_data.IterDataPipe[dict[str, Any]]):
    def __init__(
        self: Self,
        source_datapipe: torch_data.IterDataPipe[str],
        *,
        columns: Iterable[str] | None = None,
        filter_expr: ds.Expression | None = None,
        batch_size: int = 2**10,
    ) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe
        self.columns = columns
        self.filter_expr = filter_expr
        self.batch_size = batch_size

    def pyarrow_dataset(self: Self, source: str) -> ds.Dataset:
        import pyarrow.dataset as ds

        return ds.dataset(source)

    def __len__(self: Self) -> int:
        return sum(
            self.pyarrow_dataset(source).count_rows(filter=self.filter_expr)
            for source in self.source_datapipe
        )

    def __iter__(self: Self) -> Iterator[dict[str, Any]]:
        for source in self.source_datapipe:
            dataset = self.pyarrow_dataset(source)
            for batch in dataset.to_batches(
                columns=self.columns,
                filter=self.filter_expr,
                batch_size=self.batch_size,
            ):
                yield from batch.to_pylist()


@torch_data.functional_datapipe("load_delta_table_as_dict")
class DeltaTableDictLoaderIterDataPipe(ParquetDictLoaderIterDataPipe):
    def pyarrow_dataset(self: Self, source: str) -> ds.Dataset:
        import deltalake

        return deltalake.DeltaTable(source).to_pyarrow_dataset()


@torch_data.functional_datapipe("cycle")
class CyclerIterDataPipe(torch_data.IterDataPipe[dict[str, Any]]):
    def __init__(
        self: Self,
        source_datapipe: torch_data.IterDataPipe[dict[str, Any]],
        count: int | None = None,
    ) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe
        self.count = count
        if count is not None and count < 0:
            msg = f"requires count >= 0, but {count = }"
            raise ValueError(msg)

    def __len__(self: Self) -> int:
        if self.count is None:
            # use arbitrary large number so that valid length is shown for zip
            return 2**31 - 1  # max 32-bit signed integer
        return len(self.source_datapipe) * self.count

    def __iter__(self: Self) -> Iterator[dict[str, Any]]:
        i = 0
        while self.count is None or i < self.count:
            yield from self.source_datapipe
            i += 1
