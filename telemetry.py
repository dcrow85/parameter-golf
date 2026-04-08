from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
from torch import Tensor


BYTE_TOKEN_START = 3
BYTE_TOKEN_END = 258


class TelemetryBuffer:
    """
    CPU-side telemetry accumulator for Dynamic Gravity Tokenizer experiments.

    Training enqueues per-position loss tensors with non-blocking device->host copies.
    The outer loop drains and aggregates them later so the GPU path only pays for the
    host copy request, not the Python aggregation work.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        byte_token_start: int = BYTE_TOKEN_START,
        byte_token_end: int = BYTE_TOKEN_END,
        max_pair_entries: int = 50_000,
    ):
        self.vocab_size = vocab_size
        self.byte_token_start = byte_token_start
        self.byte_token_end = byte_token_end
        self.max_pair_entries = max_pair_entries

        self.token_loss_sum = torch.zeros(vocab_size, dtype=torch.float64)
        self.token_loss_count = torch.zeros(vocab_size, dtype=torch.int64)

        self.pair_loss_sum: defaultdict[tuple[int, int], float] = defaultdict(float)
        self.pair_loss_count: defaultdict[tuple[int, int], int] = defaultdict(int)
        self.byte_run_losses: defaultdict[bytes, list[float]] = defaultdict(list)
        self.ablation_intact: defaultdict[int, list[float]] = defaultdict(list)
        self.ablation_shattered: defaultdict[int, list[float]] = defaultdict(list)

        self.pending: deque[tuple[Tensor, Tensor]] = deque()

        self.enqueue_calls = 0
        self.enqueue_ms_total = 0.0
        self.drain_calls = 0
        self.drain_ms_total = 0.0
        self.processed_batches = 0
        self.max_pending = 0

    def enqueue(self, token_ids: Tensor, loss_per_pos: Tensor) -> None:
        start = time.perf_counter()
        cpu_ids = token_ids.detach().to(device="cpu", dtype=torch.int64, non_blocking=True).contiguous()
        cpu_loss = loss_per_pos.detach().to(device="cpu", dtype=torch.float32, non_blocking=True).contiguous()
        self.pending.append((cpu_ids, cpu_loss))
        self.enqueue_calls += 1
        self.enqueue_ms_total += 1000.0 * (time.perf_counter() - start)
        self.max_pending = max(self.max_pending, len(self.pending))

    def drain(self, max_items: int | None = None) -> int:
        start = time.perf_counter()
        processed = 0
        while self.pending and (max_items is None or processed < max_items):
            token_ids, loss_per_pos = self.pending.popleft()
            self._process_batch(token_ids, loss_per_pos)
            processed += 1
            self.processed_batches += 1
        self.drain_calls += 1
        self.drain_ms_total += 1000.0 * (time.perf_counter() - start)
        return processed

    def flush(self) -> None:
        self.drain(max_items=None)

    def _process_batch(self, token_ids: Tensor, loss_per_pos: Tensor) -> None:
        flat_ids = token_ids.reshape(-1)
        flat_loss = loss_per_pos.reshape(-1).to(dtype=torch.float64)

        counts = torch.bincount(flat_ids, minlength=self.vocab_size)
        loss_sums = torch.bincount(flat_ids, weights=flat_loss, minlength=self.vocab_size)
        self.token_loss_count += counts.to(dtype=torch.int64)
        self.token_loss_sum += loss_sums.to(dtype=torch.float64)

        for row_ids, row_loss in zip(token_ids, loss_per_pos, strict=True):
            row_ids_list = row_ids.tolist()
            row_loss_list = row_loss.tolist()
            self._track_pairs(row_ids_list, row_loss_list)
            self._track_byte_runs(row_ids_list, row_loss_list)

    def _track_pairs(self, row_ids: list[int], row_loss: list[float]) -> None:
        if len(row_ids) < 2:
            return
        for prev_id, next_id, next_loss in zip(row_ids[:-1], row_ids[1:], row_loss[1:], strict=True):
            pair = (prev_id, next_id)
            if pair not in self.pair_loss_count and len(self.pair_loss_count) >= self.max_pair_entries:
                continue
            self.pair_loss_sum[pair] += float(next_loss)
            self.pair_loss_count[pair] += 1

    def _track_byte_runs(self, row_ids: list[int], row_loss: list[float]) -> None:
        i = 0
        while i < len(row_ids):
            if not self._is_byte_token(row_ids[i]):
                i += 1
                continue
            j = i
            while j < len(row_ids) and self._is_byte_token(row_ids[j]):
                j += 1
            if j - i >= 2:
                byte_string = bytes(self._token_id_to_byte_value(token_id) for token_id in row_ids[i:j])
                run_loss = float(sum(row_loss[i:j]) / (j - i))
                self.byte_run_losses[byte_string].append(run_loss)
            i = j

    def _is_byte_token(self, token_id: int) -> bool:
        return self.byte_token_start <= token_id <= self.byte_token_end

    def _token_id_to_byte_value(self, token_id: int) -> int:
        return token_id - self.byte_token_start

    def get_token_mean_loss(self, token_id: int) -> float:
        count = int(self.token_loss_count[token_id].item())
        if count == 0:
            return 0.0
        return float(self.token_loss_sum[token_id].item() / count)

    def summary(self) -> dict[str, object]:
        observed_tokens = int((self.token_loss_count > 0).sum().item())
        return {
            "vocab_size": self.vocab_size,
            "enqueue_calls": self.enqueue_calls,
            "processed_batches": self.processed_batches,
            "pending_batches": len(self.pending),
            "max_pending": self.max_pending,
            "enqueue_ms_total": self.enqueue_ms_total,
            "enqueue_ms_avg": self.enqueue_ms_total / max(self.enqueue_calls, 1),
            "drain_calls": self.drain_calls,
            "drain_ms_total": self.drain_ms_total,
            "drain_ms_avg": self.drain_ms_total / max(self.drain_calls, 1),
            "observed_tokens": observed_tokens,
            "pair_entries": len(self.pair_loss_count),
            "byte_run_entries": len(self.byte_run_losses),
            "token_loss_count_total": int(self.token_loss_count.sum().item()),
        }

    def write_summary(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
