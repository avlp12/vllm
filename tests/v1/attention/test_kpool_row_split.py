# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host admission and rank-order contract without importing CUDA kernels."""

import importlib.util
from pathlib import Path

import pytest

path = (
    Path(__file__).resolve().parents[3]
    / "vllm/model_executor/layers/kpool_row_split.py"
)
spec = importlib.util.spec_from_file_location("kpool_row_split_host", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def plan(rows=2048, world=4, rank=0, **changes):
    kwargs = dict(
        pools=8192,
        enabled=True,
        num_prefills=1,
        num_decodes=0,
        fp8_cache=True,
        capturing=False,
        local_pools=None,
    )
    kwargs.update(changes)
    return module.partition_rows(rows, world=world, rank=rank, **kwargs)


@pytest.mark.parametrize("rows", [1024, 2048, 4096])
@pytest.mark.parametrize("world", [2, 4, 8])
def test_rank_gather_preserves_absolute_query_order(rows, world):
    joined: list[int] = []
    spans = []
    for rank in range(world):
        start, stop = plan(rows, world, rank)
        spans.append(stop - start)
        # Nonzero request offsets must not be rebased by the split.
        joined.extend(range(9000 + start, 9000 + stop))
    assert joined == list(range(9000, 9000 + rows))
    assert len(set(spans)) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"enabled": False},
        {"world": 1},
        {"num_prefills": 2},
        {"num_prefills": 0},
        {"num_decodes": 1},
        {"fp8_cache": False},
        {"capturing": True},
        {"local_pools": 4096},
        {"rows": 1000},
        {"rows": 1025},
        {"pools": 4095},
    ],
)
def test_unsupported_batches_keep_the_original_path(changes):
    assert plan(**changes) is None


@pytest.mark.parametrize(
    "changes",
    [{"world": 0}, {"rank": -1}, {"rank": 4}, {"rows": -1}, {"pools": -1}],
)
def test_invalid_partition_rejected(changes):
    with pytest.raises(ValueError):
        plan(**changes)
