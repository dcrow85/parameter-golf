from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from telemetry import BYTE_TOKEN_END, BYTE_TOKEN_START


FIRST_DYNAMIC_TOKEN_ID = 259
QUIESCENT_CLASS = "QUIESCENT"
QUIESCENT_ALIASES = frozenset({"GRADUATED", QUIESCENT_CLASS})

STATIC_CORE_SEED_WORDS = (
    "because", "since", "after", "when", "while", "whether", "until",
    "so", "before",
    "however", "between", "without", "through", "during", "against",
    "the", "every", "both", "either", "such", "each",
    "some", "many", "most", "more", "other", "another",
    "would", "could", "should", "also", "still", "already",
    "their", "which", "where", "what", "who", "how", "that",
    "under", "within", "about", "from", "into", "over", "upon",
    "rather", "quite", "unless",
)


def normalize_classification(classification: str | None) -> str:
    if classification in QUIESCENT_ALIASES:
        return QUIESCENT_CLASS
    return str(classification or "")


def is_quiescent_classification(classification: str | None) -> bool:
    return normalize_classification(classification) == QUIESCENT_CLASS


def resolve_static_core_ids(vocabulary_json_path: str | Path, seed_words: tuple[str, ...] = STATIC_CORE_SEED_WORDS) -> set[int]:
    """
    Resolve the explicit closed-class seed words against the static gravity vocabulary.

    The build pipeline preserves token ordering, so merge token id = 259 + index within
    vocabulary_beta_*.json["tokens"].
    """
    path = Path(vocabulary_json_path)
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    targets = {word.lower() for word in seed_words}
    resolved: set[int] = set()
    for offset, token in enumerate(data.get("tokens", []), start=FIRST_DYNAMIC_TOKEN_ID):
        readable = str(token.get("readable", "")).strip().lower()
        piece = str(token.get("piece", "")).lstrip("\u2581").strip().lower()
        if readable in targets or piece in targets:
            resolved.add(offset)
    return resolved


def _load_vocab_metadata(vocabulary_json_path: str | Path) -> dict[int, dict[str, Any]]:
    path = Path(vocabulary_json_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata: dict[int, dict[str, Any]] = {}
    for offset, token in enumerate(data.get("tokens", []), start=FIRST_DYNAMIC_TOKEN_ID):
        metadata[offset] = {
            "piece": token.get("piece"),
            "readable": token.get("readable"),
            "token_bytes": token.get("token_bytes"),
            "ablation_leverage": token.get("ablation_leverage"),
            "score": token.get("score"),
            "corpus_frequency": token.get("corpus_frequency"),
        }
    return metadata


def _quantile(values: list[float], q: float, default: float = 0.0) -> float:
    if not values:
        return default
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, q).item())


def calibrate_velocity_thresholds(
    tracker: "VelocityTracker",
    static_core_ids: set[int] | None = None,
    min_observations: int = 500,
    panic_quantile: float = 0.75,
    work_quantile: float = 0.75,
    work_floor_quantile: float = 0.25,
) -> dict[str, float | int]:
    static_ids = static_core_ids or set()
    required_obs = max(int(min_observations), int(tracker.min_obs_floor))
    panic_values: list[float] = []
    work_values: list[float] = []
    for token_id in range(FIRST_DYNAMIC_TOKEN_ID, tracker.vocab_size):
        if token_id in static_ids:
            continue
        if not tracker.is_mature(token_id):
            continue
        if int(tracker.observation_count[token_id].item()) < required_obs:
            continue
        panic_values.append(float(tracker.panic_ratio[token_id].item()))
        work_values.append(float(tracker.active_work[token_id].item()))

    if not panic_values or not work_values:
        return {
            "panic_threshold": 0.0,
            "work_threshold": 0.0,
            "work_floor": 0.0,
            "calibration_tokens": 0,
        }

    work_threshold = _quantile(work_values, work_quantile)
    work_floor = _quantile(work_values, work_floor_quantile)
    return {
        "panic_threshold": _quantile(panic_values, panic_quantile),
        "work_threshold": work_threshold,
        "work_floor": min(work_floor, work_threshold),
        "calibration_tokens": len(work_values),
    }


def classify_token(
    tracker: "VelocityTracker",
    token_id: int,
    static_core_ids: set[int] | None = None,
    min_observations: int = 500,
    panic_threshold: float = 0.0,
    work_threshold: float = 0.0,
    work_floor: float = 0.0,
) -> str:
    static_ids = static_core_ids or set()
    required_obs = max(int(min_observations), int(tracker.min_obs_floor))
    if token_id < FIRST_DYNAMIC_TOKEN_ID:
        return "NON_DYNAMIC"
    if token_id in static_ids:
        return "STATIC"

    count = int(tracker.observation_count[token_id].item())
    if not tracker.is_mature(token_id):
        return "IMMATURE"
    if count < required_obs:
        return "IMMATURE"

    panic_ratio = float(tracker.panic_ratio[token_id].item())
    active_work = float(tracker.active_work[token_id].item())
    if panic_ratio > panic_threshold and active_work > work_threshold:
        return "DEBRIS"
    if panic_ratio < panic_threshold and active_work < work_floor:
        return QUIESCENT_CLASS
    if panic_ratio < panic_threshold and active_work > work_threshold:
        return "CRYSTAL"
    if panic_ratio > panic_threshold and active_work < work_floor:
        return "DEBRIS"
    return "MARGINAL"


def build_phase_space_snapshot(
    tracker: "VelocityTracker",
    vocabulary_json_path: str | Path | None = None,
    static_core_ids: set[int] | None = None,
    min_observations: int = 500,
    panic_threshold: float | None = None,
    work_threshold: float | None = None,
    work_floor: float | None = None,
    panic_quantile: float = 0.75,
    work_quantile: float = 0.75,
    work_floor_quantile: float = 0.25,
) -> dict[str, Any]:
    static_ids = static_core_ids or set()
    thresholds = calibrate_velocity_thresholds(
        tracker,
        static_core_ids=static_ids,
        min_observations=min_observations,
        panic_quantile=panic_quantile,
        work_quantile=work_quantile,
        work_floor_quantile=work_floor_quantile,
    )
    panic_cut = thresholds["panic_threshold"] if panic_threshold is None else panic_threshold
    work_cut = thresholds["work_threshold"] if work_threshold is None else work_threshold
    work_floor_cut = thresholds["work_floor"] if work_floor is None else work_floor

    metadata = _load_vocab_metadata(vocabulary_json_path) if vocabulary_json_path is not None else {}
    rows: list[dict[str, Any]] = []
    class_counts: defaultdict[str, int] = defaultdict(int)
    for token_id in range(FIRST_DYNAMIC_TOKEN_ID, tracker.vocab_size):
        row = {
            "token_id": token_id,
            "panic_ratio": float(tracker.panic_ratio[token_id].item()),
            "active_work": float(tracker.active_work[token_id].item()),
            "panic_cv": float(tracker.get_panic_cv(token_id)),
            "work_cv": float(tracker.get_work_cv(token_id)),
            "observation_count": int(tracker.observation_count[token_id].item()),
            "is_stable": bool(tracker.is_stable[token_id].item()),
            "is_static_core": token_id in static_ids,
            "classification": classify_token(
                tracker,
                token_id,
                static_core_ids=static_ids,
                min_observations=min_observations,
                panic_threshold=float(panic_cut),
                work_threshold=float(work_cut),
                work_floor=float(work_floor_cut),
            ),
        }
        row.update(metadata.get(token_id, {}))
        rows.append(row)
        class_counts[row["classification"]] += 1

    return {
        "thresholds": {
            "panic_threshold": float(panic_cut),
            "work_threshold": float(work_cut),
            "work_floor": float(work_floor_cut),
            "min_observations": int(min_observations),
            "min_obs_floor": int(tracker.min_obs_floor),
            "stability_epsilon": float(tracker.stability_epsilon),
            "panic_quantile": float(panic_quantile),
            "work_quantile": float(work_quantile),
            "work_floor_quantile": float(work_floor_quantile),
            "calibration_tokens": int(thresholds["calibration_tokens"]),
            "stable_tokens": int(tracker.is_stable.sum().item()),
        },
        "static_core_seed_words": list(STATIC_CORE_SEED_WORDS),
        "static_core_ids": sorted(static_ids),
        "class_counts": dict(sorted(class_counts.items())),
        "rows": rows,
    }


def write_phase_space_snapshot(
    path: str | Path,
    tracker: "VelocityTracker",
    vocabulary_json_path: str | Path | None = None,
    static_core_ids: set[int] | None = None,
    min_observations: int = 500,
    panic_threshold: float | None = None,
    work_threshold: float | None = None,
    work_floor: float | None = None,
    panic_quantile: float = 0.75,
    work_quantile: float = 0.75,
    work_floor_quantile: float = 0.25,
) -> dict[str, Any]:
    snapshot = build_phase_space_snapshot(
        tracker,
        vocabulary_json_path=vocabulary_json_path,
        static_core_ids=static_core_ids,
        min_observations=min_observations,
        panic_threshold=panic_threshold,
        work_threshold=work_threshold,
        work_floor=work_floor,
        panic_quantile=panic_quantile,
        work_quantile=work_quantile,
        work_floor_quantile=work_floor_quantile,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    return snapshot


class PanicPredictions:
    """
    Captures the model's top-k output predictions at positions where
    a token triggers a high instantaneous panic ratio.

    Used by the fusion targeting logic to determine what the model
    is trying to produce when it panics at DEBRIS tokens.
    """

    def __init__(
        self,
        top_k: int = 5,
        panic_threshold: float = 2.0,
        min_token_id: int = FIRST_DYNAMIC_TOKEN_ID,
    ):
        self.top_k = top_k
        self.panic_threshold = panic_threshold
        self.min_token_id = min_token_id
        self.predictions: defaultdict[int, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.panic_event_count: defaultdict[int, int] = defaultdict(int)

    def record(self, token_ids: Tensor, velocity_map: Tensor, logits: Tensor) -> int:
        """
        Record top-k predictions at positions with high instantaneous panic.

        token_ids: [batch, seq_len]
        velocity_map: [n_layers, batch, seq_len]
        logits: [batch, seq_len, vocab_size]

        Returns the number of panic events recorded.
        """
        mid_layer = min(5, velocity_map.shape[0] - 1)
        l_mid = velocity_map[mid_layer]
        l_last = velocity_map[-1]
        instant_panic = l_last / l_mid.clamp(min=1e-6)

        mask = (instant_panic > self.panic_threshold) & (token_ids >= self.min_token_id)
        # Exclude the last position (no next-token prediction available)
        mask[:, -1] = False

        if not mask.any():
            return 0

        batch_idx, seq_idx = torch.where(mask)
        panic_tids = token_ids[batch_idx, seq_idx].tolist()
        # Top-k predictions at the SAME position (model's output distribution when panicking)
        panic_logits = logits[batch_idx, seq_idx]
        topk_ids = panic_logits.topk(self.top_k, dim=-1).indices.tolist()

        events = 0
        for tid, top_preds in zip(panic_tids, topk_ids):
            for pred_id in top_preds:
                self.predictions[tid][pred_id] += 1
            self.panic_event_count[tid] += 1
            events += 1
        return events

    def get_dominant_continuation(
        self, token_id: int, min_events: int = 10, dominance_threshold: float = 0.6
    ) -> tuple[int, float] | None:
        if self.panic_event_count[token_id] < min_events:
            return None
        preds = self.predictions[token_id]
        total = sum(preds.values())
        if total == 0:
            return None
        top_pred = max(preds, key=lambda k: preds[k])
        fraction = preds[top_pred] / total
        if fraction >= dominance_threshold:
            return (top_pred, fraction)
        return None

    def snapshot(self) -> dict[str, Any]:
        rows = []
        for tid in sorted(self.panic_event_count):
            preds = self.predictions[tid]
            total = sum(preds.values())
            top_5 = sorted(preds.items(), key=lambda kv: -kv[1])[:5]
            rows.append({
                "token_id": tid,
                "panic_events": self.panic_event_count[tid],
                "top_predictions": [
                    {"pred_id": pid, "count": cnt, "fraction": cnt / total if total else 0}
                    for pid, cnt in top_5
                ],
            })
        return {
            "panic_threshold": self.panic_threshold,
            "top_k": self.top_k,
            "total_tokens_tracked": len(self.panic_event_count),
            "total_panic_events": sum(self.panic_event_count.values()),
            "tokens": rows,
        }

    def reset(self) -> None:
        self.predictions.clear()
        self.panic_event_count.clear()


class VelocityTracker:
    """
    Aggregates Panic Ratio and Active Work from residual velocity snapshots.

    Snapshots are enqueued with non-blocking host copies and drained later on CPU.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        ema_alpha: float = 0.3,
        stability_epsilon: float = 0.05,
        min_obs_floor: int = 30,
        byte_token_start: int = BYTE_TOKEN_START,
        byte_token_end: int = BYTE_TOKEN_END,
    ):
        self.vocab_size = vocab_size
        self.ema_alpha = ema_alpha
        self.stability_epsilon = stability_epsilon
        self.min_obs_floor = min_obs_floor
        self.byte_token_start = byte_token_start
        self.byte_token_end = byte_token_end

        self.panic_ratio = torch.zeros(vocab_size, dtype=torch.float32)
        self.active_work = torch.zeros(vocab_size, dtype=torch.float32)
        self.panic_ratio_var = torch.zeros(vocab_size, dtype=torch.float32)
        self.active_work_var = torch.zeros(vocab_size, dtype=torch.float32)
        self.observation_count = torch.zeros(vocab_size, dtype=torch.int64)
        self.is_stable = torch.zeros(vocab_size, dtype=torch.bool)

        self.byte_run_panic_sum: defaultdict[bytes, float] = defaultdict(float)
        self.byte_run_panic_count: defaultdict[bytes, int] = defaultdict(int)

        self.pending: deque[tuple[Tensor, Tensor]] = deque()
        self.enqueue_calls = 0
        self.enqueue_ms_total = 0.0
        self.drain_calls = 0
        self.drain_ms_total = 0.0
        self.processed_snapshots = 0
        self.max_pending = 0

    def enqueue(self, token_ids: Tensor, velocity_map: Tensor) -> None:
        start = time.perf_counter()
        cpu_ids = token_ids.detach().to(device="cpu", dtype=torch.int64, non_blocking=True).contiguous()
        cpu_velocity = velocity_map.detach().to(device="cpu", dtype=torch.float32, non_blocking=True).contiguous()
        self.pending.append((cpu_ids, cpu_velocity))
        self.enqueue_calls += 1
        self.enqueue_ms_total += 1000.0 * (time.perf_counter() - start)
        self.max_pending = max(self.max_pending, len(self.pending))

    def drain(self, max_items: int | None = None) -> int:
        start = time.perf_counter()
        processed = 0
        while self.pending and (max_items is None or processed < max_items):
            token_ids, velocity_map = self.pending.popleft()
            self.update(token_ids, velocity_map)
            processed += 1
            self.processed_snapshots += 1
        self.drain_calls += 1
        self.drain_ms_total += 1000.0 * (time.perf_counter() - start)
        return processed

    def flush(self) -> None:
        self.drain(max_items=None)

    def update(self, token_ids: Tensor, velocity_map: Tensor) -> None:
        if velocity_map.ndim != 3:
            raise ValueError(f"velocity_map must be [layers, batch, seq], got {tuple(velocity_map.shape)}")
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be [batch, seq], got {tuple(token_ids.shape)}")

        layer_count = velocity_map.shape[0]
        mid_layer = min(5, layer_count - 1)
        l5_vel = velocity_map[mid_layer]
        l_last_vel = velocity_map[-1]
        total_vel = velocity_map.sum(dim=0)

        flat_ids = token_ids.reshape(-1)
        flat_panic_num = l_last_vel.reshape(-1)
        flat_panic_den = l5_vel.reshape(-1).clamp(min=1e-6)
        flat_work = total_vel.reshape(-1)

        for tid in flat_ids.unique(sorted=False):
            token_id = int(tid.item())
            if token_id < FIRST_DYNAMIC_TOKEN_ID:
                continue
            mask = flat_ids == token_id
            observations_added = int(mask.sum().item())
            mean_panic = float((flat_panic_num[mask] / flat_panic_den[mask]).mean().item())
            mean_work = float(flat_work[mask].mean().item())

            prior_count = int(self.observation_count[token_id].item())
            if prior_count == 0:
                self.panic_ratio[token_id] = mean_panic
                self.active_work[token_id] = mean_work
            else:
                a = self.ema_alpha
                old_panic = float(self.panic_ratio[token_id].item())
                old_work = float(self.active_work[token_id].item())
                self.panic_ratio[token_id] = a * mean_panic + (1.0 - a) * old_panic
                self.active_work[token_id] = a * mean_work + (1.0 - a) * old_work

                panic_dev_sq = (mean_panic - old_panic) ** 2
                work_dev_sq = (mean_work - old_work) ** 2
                self.panic_ratio_var[token_id] = a * panic_dev_sq + (1.0 - a) * self.panic_ratio_var[token_id]
                self.active_work_var[token_id] = a * work_dev_sq + (1.0 - a) * self.active_work_var[token_id]

            self.observation_count[token_id] += observations_added

            if (
                not bool(self.is_stable[token_id].item())
                and int(self.observation_count[token_id].item()) >= self.min_obs_floor
            ):
                panic_cv = self.get_panic_cv(token_id)
                work_cv = self.get_work_cv(token_id)
                if panic_cv < self.stability_epsilon and work_cv < self.stability_epsilon:
                    self.is_stable[token_id] = True

        for row_ids, row_panic in zip(token_ids, l_last_vel / l5_vel.clamp(min=1e-6), strict=True):
            self._track_byte_run_panic(row_ids.tolist(), row_panic.tolist())

    def _track_byte_run_panic(self, row_ids: list[int], row_panic: list[float]) -> None:
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
                mean_panic = float(sum(row_panic[i:j]) / (j - i))
                self.byte_run_panic_sum[byte_string] += mean_panic
                self.byte_run_panic_count[byte_string] += 1
            i = j

    def _is_byte_token(self, token_id: int) -> bool:
        return self.byte_token_start <= token_id <= self.byte_token_end

    def _token_id_to_byte_value(self, token_id: int) -> int:
        return token_id - self.byte_token_start

    def get_byte_run_panic(self, byte_string: bytes) -> float:
        count = self.byte_run_panic_count.get(byte_string, 0)
        if count == 0:
            return 0.0
        return self.byte_run_panic_sum[byte_string] / count

    def get_panic_cv(self, token_id: int) -> float:
        mean_value = abs(float(self.panic_ratio[token_id].item()))
        if mean_value <= 1e-6:
            return 0.0
        return float(self.panic_ratio_var[token_id].item() ** 0.5) / mean_value

    def get_work_cv(self, token_id: int) -> float:
        mean_value = abs(float(self.active_work[token_id].item()))
        if mean_value <= 1e-6:
            return 0.0
        return float(self.active_work_var[token_id].item() ** 0.5) / mean_value

    def is_mature(self, token_id: int) -> bool:
        if token_id < FIRST_DYNAMIC_TOKEN_ID:
            return False
        return bool(self.is_stable[token_id].item())

    def summary(self) -> dict[str, object]:
        observed_tokens = int((self.observation_count > 0).sum().item())
        return {
            "vocab_size": self.vocab_size,
            "stability_epsilon": self.stability_epsilon,
            "min_obs_floor": self.min_obs_floor,
            "enqueue_calls": self.enqueue_calls,
            "processed_snapshots": self.processed_snapshots,
            "pending_snapshots": len(self.pending),
            "max_pending": self.max_pending,
            "enqueue_ms_total": self.enqueue_ms_total,
            "enqueue_ms_avg": self.enqueue_ms_total / max(self.enqueue_calls, 1),
            "drain_calls": self.drain_calls,
            "drain_ms_total": self.drain_ms_total,
            "drain_ms_avg": self.drain_ms_total / max(self.drain_calls, 1),
            "observed_tokens": observed_tokens,
            "stable_tokens": int(self.is_stable.sum().item()),
            "observed_positions_total": int(self.observation_count.sum().item()),
            "byte_run_panic_entries": len(self.byte_run_panic_count),
        }

    def write_summary(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
