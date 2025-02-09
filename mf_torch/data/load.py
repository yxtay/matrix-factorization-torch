from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.utils.data as torch_data
import torch.utils.data._utils.collate as torch_collate

from mf_torch.params import NUM_EMBEDDINGS, NUM_HASHES, PADDING_IDX

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Self

    import pyarrow.dataset as ds


def collate_features(
    value: None | str | int | dict | list,
    key: str = "",
    feature_names: dict[str, str] | None = None,
) -> tuple[list[str], torch.Tensor]:
    import itertools

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
        values, weights = zip(*[collate_features(v, key, feature_names) for v in value])
    elif isinstance(value, dict):
        # nested feature
        key_fmt = f"{key}:{{}}" if key else "{}"
        values, weights = zip(
            *[
                collate_features(v, key_fmt.format(feature_names[k]), feature_names)
                for k, v in value.items()
            ]
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
    import xxhash

    features = select_fields(row, fields=feature_names)
    feature_values, feature_weights = collate_features(
        features, feature_names=feature_names
    )
    feature_hashes, feature_weights = hash_features(
        feature_values,
        feature_weights,
        num_hashes=num_hashes,
        num_embeddings=num_embeddings,
    )
    processed = {
        "idx": xxhash.xxh32_intdigest(str(row[idx])),
        "feature_values": feature_values,
        "feature_hashes": feature_hashes,
        "feature_weights": feature_weights,
    }
    row.update({f"{prefix}{key}": value for key, value in processed.items()})
    return row


def score_interactions(
    row: dict, *, label: str = "label", weight: str = "weight"
) -> dict:
    row["label"] = row[label] or 0
    row["weight"] = abs(row[weight] or 0)
    return row


def select_fields(row: dict, *, fields: list[str]) -> dict:
    return {key: row[key] for key in fields}


def merge_rows(rows: Iterable[dict]) -> dict:
    new_row = {}
    for row in rows:
        new_row |= row
    return new_row


def pad_jagged_tensors(
    tensors: list[torch.Tensor], padding: int = PADDING_IDX
) -> torch.Tensor:
    return torch.nested.nested_tensor(tensors).to_padded_tensor(padding=padding)


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
        return pad_jagged_tensors(batch)
    return torch_collate.collate_tensor_fn(batch, collate_fn_map=collate_fn_map)


torch_collate.default_collate_fn_map[torch.Tensor] = collate_tensor_fn


@torch_data.functional_datapipe("load_parquet_as_dict")
class ParquetDictLoaderIterDataPipe(torch_data.IterDataPipe):
    def __init__(
        self: Self,
        source_datapipe: torch_data.IterDataPipe,
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
        source_datapipe: torch_data.IterDataPipe,
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
