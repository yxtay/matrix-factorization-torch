from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.utils.data as torch_data
import torch.utils.data._utils.collate as torch_collate

if TYPE_CHECKING:
    import numpy as np
    import pyarrow.dataset as ds


def hash_features(
    row: dict[str, int | float | str | list[int, str]],
    *,
    idx: str,
    feature_names: list[str],
    num_hashes: int = 2,
    num_buckets: int = 2**16 + 1,
    out_prefix: str = "",
    keep_input: bool = True,
) -> torch.Tensor:
    import mmh3

    feature_values = []
    feature_weights = []
    num_features = 0
    # categorical features
    cat_features = [
        f"{key}:{row[key]}" for key in feature_names if isinstance(row[key], (int, str))
    ]
    if len(cat_features) > 0:
        feature_values.extend(cat_features)
        feature_weights.append(torch.ones(len(cat_features)))
        num_features += len(cat_features)

    # float features
    float_features = [key for key in feature_names if isinstance(row[key], float)]
    if len(float_features) > 0:
        feature_values.extend(float_features)
        feature_weights.append(torch.as_tensor([row[key] for key in float_features]))
        num_features += len(float_features)

    # multi categorical features
    for key in feature_names:
        if isinstance(row[key], Iterable) and not isinstance(row[key], str):
            iter_feature_values = [
                f"{key}:{value}" for value in row[key] if isinstance(value, (int, str))
            ]
            num_values = len(iter_feature_values)
            if num_values == 0:
                continue
            iter_feature_weights = torch.ones(num_values) / num_values

            feature_values.extend(iter_feature_values)
            feature_weights.append(iter_feature_weights)
            num_features += 1

    feature_hashes = [
        mmh3.hash(values, seed)
        for seed in range(num_hashes)
        for values in feature_values
    ]
    feature_hashes = torch.as_tensor(feature_hashes) % (num_buckets - 1) + 1
    feature_weights = (
        torch.cat(feature_weights).tile(num_hashes) / num_features / num_hashes
    )

    new_row = {
        f"{out_prefix}idx": row[idx] or 0,
        f"{out_prefix}feature_hashes": feature_hashes.to(torch.int32),
        f"{out_prefix}feature_weights": feature_weights.to(torch.float32),
    }
    if not keep_input:
        return new_row
    row.update(new_row)
    return row


def gather_inputs(
    row: dict[str, int | float | list[int | float] | None],
    *,
    label: str = "label",
    weight: str = "weight",
) -> dict[str, int | float | list[int | float]]:
    label_value = row[label] or 0
    inputs = {
        "label": bool(label_value > 0) - bool(label_value < 0),
        "weight": row[weight] or 0,
        "user_idx": row["user_idx"],
        "user_feature_hashes": row["user_feature_hashes"],
        "user_feature_weights": row["user_feature_weights"],
        "item_idx": row["item_idx"],
        "item_feature_hashes": row["item_feature_hashes"],
        "item_feature_weights": row["item_feature_weights"],
    }
    return inputs


def merge_rows(rows):
    new_row = {**rows[0]}
    for row in rows[1:]:
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
        nested = torch.nested.as_nested_tensor(batch)
        return torch.nested.to_padded_tensor(nested, padding=0)
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


@torch_data.functional_datapipe("load_pyarrow_dataset_as_dict")
class PyArrowDatasetDictLoaderIterDataPipe(torch_data.IterDataPipe):
    def __init__(
        self,
        source_datapipe: Iterable[str | Path],
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

    def pyarrow_dataset(self, source: str | Path) -> ds.Dataset:
        import pyarrow.dataset as ds

        return ds.dataset(source)

    def __len__(self) -> int:
        num_rows = sum(
            self.pyarrow_dataset(source).count_rows(filter=self.filter_expr)
            for source in self.source_datapipe
        )
        return num_rows

    def __iter__(self) -> Iterable[dict]:
        for source in self.source_datapipe:
            dataset = self.pyarrow_dataset(source)
            for batch in dataset.to_batches(
                columns=self.columns,
                filter=self.filter_expr,
                batch_size=self.batch_size,
            ):
                yield from batch.to_pylist()


@torch_data.functional_datapipe("load_delta_table_as_dict")
class DeltaTableDictLoaderIterDataPipe(PyArrowDatasetDictLoaderIterDataPipe):
    def pyarrow_dataset(self, source: str | Path) -> ds.Dataset:
        import deltalake as dl

        return dl.DeltaTable(source).to_pyarrow_dataset()


@torch_data.functional_datapipe("cycle")
class CyclerIterDataPipe(torch_data.IterDataPipe):
    def __init__(
        self,
        source_datapipe: Iterable[str | Path],
        count: int | None = None,
    ) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe
        self.count = count
        if count is not None and count < 0:
            msg = f"requires count >= 0, but {count = }"
            raise ValueError(msg)

    def __len__(self) -> int:
        if self.count is None:
            # use arbitrary large number so that valid length is shown for zip
            return 2**31 - 1  # max 32-bit signed integer
        return len(self.source_datapipe) * self.count

    def __iter__(self) -> Iterable[dict]:
        i = 0
        while self.count is None or i < self.count:
            yield from self.source_datapipe
            i += 1
