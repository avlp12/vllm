# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in query-row sharding for a single eager kpool prefill."""

import os

_group = None
_enabled = False


def initialize_row_split() -> None:
    global _group, _enabled
    import torch.distributed as dist

    from vllm.distributed import get_tp_group

    group = get_tp_group()
    if _group is group:
        return
    value = os.environ.get("VLLM_GLM_KPOOL_ROW_SPLIT", "0")
    values = [None] * group.world_size
    dist.all_gather_object(values, value, group=group.cpu_group)
    if value not in ("0", "1") or any(v != value for v in values):
        raise ValueError("VLLM_GLM_KPOOL_ROW_SPLIT must agree on every rank: 0 or 1")
    _group, _enabled = group, value == "1"


def partition_rows(
    rows: int,
    pools: int,
    world: int,
    rank: int,
    *,
    enabled: bool,
    num_prefills: int,
    num_decodes: int,
    fp8_cache: bool,
    capturing: bool,
    local_pools: int | None,
) -> tuple[int, int] | None:
    if world < 1 or not 0 <= rank < world or rows < 0 or pools < 0:
        raise ValueError("invalid row partition")
    if (
        not enabled
        or world == 1
        or num_prefills != 1
        or num_decodes != 0
        or not fp8_cache
        or capturing
        or (local_pools is not None and local_pools != pools)
        or rows < 1024
        or pools < 4096
        or rows % (world * 64)
    ):
        return None
    extent = rows // world
    return rank * extent, (rank + 1) * extent


def plan(chunk, metadata, *, fp8_cache: bool) -> tuple[int, int] | None:
    if not _enabled:
        return None
    import torch

    assert _group is not None, "row-split group is not initialized"
    return partition_rows(
        chunk.token_end - chunk.token_start,
        chunk.total_seq_lens,
        _group.world_size,
        _group.rank_in_group,
        enabled=_enabled,
        num_prefills=metadata.num_prefills,
        num_decodes=metadata.num_decodes,
        fp8_cache=fp8_cache,
        capturing=torch.cuda.is_current_stream_capturing(),
        local_pools=getattr(chunk, "local_total_seq_lens", None),
    )


def gather(local):
    assert _group is not None, "row-split group is not initialized"
    return _group.all_gather(local.contiguous(), dim=0)
