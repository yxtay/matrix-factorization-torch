from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.utils.data as torch_data
import torch.utils.data._utils.collate as torch_collate

from mf_torch.params import NUM_EMBEDDINGS, NUM_HASHES, PADDING_IDX

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Iterator, Self

    import numpy as np
    import pyarrow.dataset as ds


def collate_features(
    value: None | str | float | dict | list, key: str = ""
) -> tuple[list[str], torch.Tensor]:
    import itertools

    if value is None:
        # ignore null values
        return [], torch.zeros(0)

    if isinstance(value, (str | int)):
        # category feature
        return [f"{key}:{value}"], torch.ones(1)

    if isinstance(value, float):
        # float feature
        return [key], torch.as_tensor([value])

    if isinstance(value, list):
        # list feature, potentially nested
        values, weights = zip(*[collate_features(v, key) for v in value])
    elif isinstance(value, dict):
        # nested feature
        key_fmt = f"{key}:{{}}" if key else "{}"
        values, weights = zip(
            *[collate_features(v, key_fmt.format(k)) for k, v in value.items()]
        )

    values = list(itertools.chain(*values))
    weights = torch.concatenate(weights) / len(value)
    return values, weights


def hash_features(
    values: list[str],
    weights: torch.Tensor,
    *,
    num_hashes: int = NUM_HASHES,
    num_embeddings: int = NUM_EMBEDDINGS,
) -> tuple[torch.Tensor, torch.Tensor]:
    import mmh3

    hashes = [mmh3.hash(val, seed) for val in values for seed in range(num_hashes)]
    hashes = torch.as_tensor(hashes) % (num_embeddings - 1) + 1
    weights = weights.repeat_interleave(num_hashes) / num_hashes
    return hashes, weights


def process_features(
    row: dict,
    *,
    idx: str,
    feature_names: list[str],
    prefix: str = "",
    num_hashes: int = NUM_HASHES,
    num_embeddings: int = NUM_EMBEDDINGS,
) -> dict:
    import mmh3

    features = select_fields(row, fields=feature_names)
    feature_values, feature_weights = collate_features(features)
    feature_hashes, feature_weights = hash_features(
        feature_values,
        feature_weights,
        num_hashes=num_hashes,
        num_embeddings=num_embeddings,
    )
    return {
        **row,
        f"{prefix}idx": mmh3.hash(str(row[idx])),
        f"{prefix}feature_values": feature_values,
        f"{prefix}feature_hashes": feature_hashes,
        f"{prefix}feature_weights": feature_weights,
    }


def score_interactions(
    row: dict, *, label: str = "label", weight: str = "weight"
) -> dict:
    label_value = row[label] or 0
    return {
        **row,
        "label": bool(label_value > 0) - bool(label_value < 0),
        "weight": row[weight] or 0,
    }


def select_fields(row: dict, *, fields: list[str]) -> dict:
    return {key: row[key] for key in fields}


def merge_rows(rows: Iterable[dict]) -> dict:
    new_row = {}
    for row in rows:
        new_row = {**new_row, **row}
    return new_row


def collate_tensor_fn(
    batch: Iterable[torch.Tensor], *, collate_fn_map: dict | None = None
) -> torch.Tensor:
    it = iter(batch)
    elem_size = next(it).size()
    if (
        any(elem.size() != elem_size for elem in it)
        and (
            # only last dimension different
            all(elem.size()[:-1] == elem_size[:-1] for elem in it)
            # only first dimension differen
            or all(elem.size()[1:] == elem_size[1:] for elem in it)
        )
    ):
        # pad tensor if only first or last dimensions are different
        return torch.nested.as_nested_tensor(batch).to_padded_tensor(
            padding=PADDING_IDX
        )
    return torch_collate.collate_tensor_fn(batch, collate_fn_map=collate_fn_map)


torch_collate.default_collate_fn_map[torch.Tensor] = collate_tensor_fn


def ray_collate_fn(
    batch: np.ndarray | dict[str, np.ndarray],
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(batch, dict):
        return {
            col_name: torch_data.default_collate(col_batch)
            for col_name, col_batch in batch.items()
        }

    return torch_data.default_collate(batch)


@torch_data.functional_datapipe("load_parquet_as_dict")
class ParquetDictLoaderIterDataPipe(torch_data.IterDataPipe):
    def __init__(
        self: Self,
        source_datapipe: Iterable[str],
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

    def __iter__(self: Self) -> Iterator[dict]:
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
class CyclerIterDataPipe(torch_data.IterDataPipe):
    def __init__(
        self: Self,
        source_datapipe: Iterable[dict],
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

    def __iter__(self: Self) -> Iterator[dict]:
        i = 0
        while self.count is None or i < self.count:
            yield from self.source_datapipe
            i += 1
