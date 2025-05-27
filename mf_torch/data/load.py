from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.utils.data as torch_data

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import pyarrow.dataset as ds


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
        "embedding": model([example["text"]]).squeeze(0).numpy(force=True),
    }


@torch_data.functional_datapipe("load_parquet_as_dict")
class ParquetDictLoaderIterDataPipe(torch_data.IterDataPipe[dict[str, Any]]):
    def __init__(
        self,
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

    def pyarrow_dataset(self, source: str) -> ds.Dataset:
        import pyarrow.dataset as ds

        return ds.dataset(source)

    def __len__(self) -> int:
        return sum(
            self.pyarrow_dataset(source).count_rows(filter=self.filter_expr)
            for source in self.source_datapipe
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for source in self.source_datapipe:
            dataset = self.pyarrow_dataset(source)
            for batch in dataset.to_batches(
                columns=self.columns,
                filter=self.filter_expr,
                batch_size=self.batch_size,
            ):
                yield from batch.to_pylist()


@torch_data.functional_datapipe("cycle")
class CyclerIterDataPipe(torch_data.IterDataPipe[dict[str, Any]]):
    def __init__(
        self,
        source_datapipe: torch_data.IterDataPipe[dict[str, Any]],
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

    def __iter__(self) -> Iterator[dict[str, Any]]:
        i = 0
        while self.count is None or i < self.count:
            yield from self.source_datapipe
            i += 1
