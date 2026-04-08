from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from velocity import FIRST_DYNAMIC_TOKEN_ID, normalize_classification


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


def _phase_classifications(phase_space_snapshot: dict[str, Any] | None) -> dict[int, str]:
    if not phase_space_snapshot:
        return {}
    rows = phase_space_snapshot.get("rows", [])
    if not isinstance(rows, list):
        return {}
    return {int(row["token_id"]): normalize_classification(row.get("classification")) for row in rows}


class GradientCoherenceTracker:
    """
    Tracks directional coherence of token embedding-row gradients over time.

    Phi(token) = ||EMA(g)|| / EMA(||g||)

    We update only for tokens that were actually present in the training step.
    This keeps the signal focused on the effective row updates for live tokens
    rather than every output-logit row in the tied embedding.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        model_dim: int = 384,
        ema_alpha: float = 0.05,
        min_updates: int = 10,
        min_token_id: int = 0,
    ):
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.ema_alpha = ema_alpha
        self.min_updates = min_updates
        self.min_token_id = min_token_id

        self.ema_grad = torch.zeros((vocab_size, model_dim), dtype=torch.float32)
        self.ema_grad_norm = torch.zeros(vocab_size, dtype=torch.float32)
        self.grad_norm_sum = torch.zeros(vocab_size, dtype=torch.float64)
        self.grad_norm_max = torch.zeros(vocab_size, dtype=torch.float32)
        self.update_count = torch.zeros(vocab_size, dtype=torch.int64)
        self.last_update_step = torch.full((vocab_size,), -1, dtype=torch.int64)

        self.update_calls = 0
        self.update_ms_total = 0.0
        self.total_rows_updated = 0

    def update(self, token_ids: Tensor, embedding_grad: Tensor, step: int) -> int:
        start = time.perf_counter()
        if token_ids.numel() == 0:
            self.update_calls += 1
            self.update_ms_total += 1000.0 * (time.perf_counter() - start)
            return 0

        token_ids = token_ids.detach().to(dtype=torch.int64)
        if self.min_token_id > 0:
            token_ids = token_ids[token_ids >= self.min_token_id]
        if token_ids.numel() == 0:
            self.update_calls += 1
            self.update_ms_total += 1000.0 * (time.perf_counter() - start)
            return 0

        unique_ids = torch.unique(token_ids, sorted=True)
        grad_rows = embedding_grad.index_select(0, unique_ids).detach().to(dtype=torch.float32)

        cpu_ids = unique_ids.to(device="cpu", dtype=torch.int64)
        cpu_grads = grad_rows.to(device="cpu", dtype=torch.float32)
        grad_norms = cpu_grads.norm(dim=-1)

        for token_id, grad_vec, grad_norm in zip(cpu_ids.tolist(), cpu_grads, grad_norms.tolist(), strict=True):
            count = int(self.update_count[token_id].item())
            if count == 0:
                self.ema_grad[token_id].copy_(grad_vec)
                self.ema_grad_norm[token_id] = float(grad_norm)
            else:
                a = self.ema_alpha
                self.ema_grad[token_id].mul_(1.0 - a).add_(grad_vec, alpha=a)
                self.ema_grad_norm[token_id] = (1.0 - a) * self.ema_grad_norm[token_id] + a * float(grad_norm)
            self.grad_norm_sum[token_id] += float(grad_norm)
            self.grad_norm_max[token_id] = max(float(self.grad_norm_max[token_id].item()), float(grad_norm))
            self.update_count[token_id] += 1
            self.last_update_step[token_id] = int(step)

        updated = int(cpu_ids.numel())
        self.update_calls += 1
        self.total_rows_updated += updated
        self.update_ms_total += 1000.0 * (time.perf_counter() - start)
        return updated

    def get_coherence(self, token_id: int) -> float:
        denom = float(self.ema_grad_norm[token_id].item())
        if denom <= 1e-12:
            return 0.0
        numer = float(self.ema_grad[token_id].norm().item())
        return numer / denom

    def get_grad_magnitude_ema(self, token_id: int) -> float:
        return float(self.ema_grad_norm[token_id].item())

    def get_grad_magnitude_mean(self, token_id: int) -> float:
        count = int(self.update_count[token_id].item())
        if count <= 0:
            return 0.0
        return float((self.grad_norm_sum[token_id] / count).item())

    def get_grad_energy_total(self, token_id: int) -> float:
        return float(self.grad_norm_sum[token_id].item())

    def get_grad_magnitude_max(self, token_id: int) -> float:
        return float(self.grad_norm_max[token_id].item())

    def summary(self) -> dict[str, Any]:
        observed_mask = self.update_count > 0
        mature_mask = self.update_count >= self.min_updates
        dynamic_mask = torch.zeros_like(observed_mask)
        dynamic_mask[self.min_token_id :] = True
        observed_dynamic = observed_mask & dynamic_mask
        mature_dynamic = mature_mask & dynamic_mask
        observed_dynamic_ids = torch.where(observed_dynamic)[0].tolist()
        mature_dynamic_ids = torch.where(mature_dynamic)[0].tolist()
        coherence_values = [self.get_coherence(i) for i in observed_dynamic_ids]
        mature_coherence_values = [self.get_coherence(i) for i in mature_dynamic_ids]
        magnitude_ema_values = [self.get_grad_magnitude_ema(i) for i in observed_dynamic_ids]
        mature_magnitude_ema_values = [self.get_grad_magnitude_ema(i) for i in mature_dynamic_ids]
        magnitude_mean_values = [self.get_grad_magnitude_mean(i) for i in observed_dynamic_ids]
        mature_magnitude_mean_values = [self.get_grad_magnitude_mean(i) for i in mature_dynamic_ids]
        high_mag_threshold = (
            float(torch.quantile(torch.tensor(mature_magnitude_ema_values, dtype=torch.float32), 0.75).item())
            if mature_magnitude_ema_values
            else 0.0
        )
        low_mag_threshold = (
            float(torch.quantile(torch.tensor(mature_magnitude_ema_values, dtype=torch.float32), 0.25).item())
            if mature_magnitude_ema_values
            else 0.0
        )
        high_mag_low_coh = 0
        high_mag_high_coh = 0
        low_mag_low_coh = 0
        for token_id in mature_dynamic_ids:
            coherence = self.get_coherence(token_id)
            magnitude = self.get_grad_magnitude_ema(token_id)
            if magnitude >= high_mag_threshold and coherence <= 0.1:
                high_mag_low_coh += 1
            if magnitude >= high_mag_threshold and coherence >= 0.8:
                high_mag_high_coh += 1
            if magnitude <= low_mag_threshold and coherence <= 0.1:
                low_mag_low_coh += 1
        return {
            "vocab_size": self.vocab_size,
            "model_dim": self.model_dim,
            "ema_alpha": self.ema_alpha,
            "min_updates": self.min_updates,
            "min_token_id": self.min_token_id,
            "update_calls": self.update_calls,
            "total_rows_updated": self.total_rows_updated,
            "update_ms_total": self.update_ms_total,
            "update_ms_avg": self.update_ms_total / max(self.update_calls, 1),
            "observed_tokens": int(observed_mask.sum().item()),
            "observed_dynamic_tokens": int(observed_dynamic.sum().item()),
            "mature_dynamic_tokens": int(mature_dynamic.sum().item()),
            "mean_dynamic_coherence": float(sum(coherence_values) / len(coherence_values)) if coherence_values else 0.0,
            "mean_mature_dynamic_coherence": float(sum(mature_coherence_values) / len(mature_coherence_values))
            if mature_coherence_values
            else 0.0,
            "mean_dynamic_grad_magnitude_ema": float(sum(magnitude_ema_values) / len(magnitude_ema_values))
            if magnitude_ema_values
            else 0.0,
            "mean_mature_dynamic_grad_magnitude_ema": float(
                sum(mature_magnitude_ema_values) / len(mature_magnitude_ema_values)
            )
            if mature_magnitude_ema_values
            else 0.0,
            "mean_dynamic_grad_magnitude_mean": float(sum(magnitude_mean_values) / len(magnitude_mean_values))
            if magnitude_mean_values
            else 0.0,
            "mean_mature_dynamic_grad_magnitude_mean": float(
                sum(mature_magnitude_mean_values) / len(mature_magnitude_mean_values)
            )
            if mature_magnitude_mean_values
            else 0.0,
            "high_coherence_dynamic_tokens": int(sum(value >= 0.8 for value in coherence_values)),
            "low_coherence_dynamic_tokens": int(sum(value <= 0.1 for value in coherence_values)),
            "high_magnitude_threshold_ema": high_mag_threshold,
            "low_magnitude_threshold_ema": low_mag_threshold,
            "high_magnitude_low_coherence_tokens": high_mag_low_coh,
            "high_magnitude_high_coherence_tokens": high_mag_high_coh,
            "low_magnitude_low_coherence_tokens": low_mag_low_coh,
            "measurement_surface": "embedding_row_grad_for_tokens_seen_in_step",
        }

    def write_summary(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")


def build_coherence_snapshot(
    tracker: GradientCoherenceTracker,
    *,
    vocabulary_json_path: str | Path | None = None,
    phase_space_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    vocab_metadata = _load_vocab_metadata(vocabulary_json_path) if vocabulary_json_path is not None else {}
    phase_class_by_token = _phase_classifications(phase_space_snapshot)
    summary = tracker.summary()
    high_mag_threshold = float(summary["high_magnitude_threshold_ema"])
    low_mag_threshold = float(summary["low_magnitude_threshold_ema"])

    rows: list[dict[str, Any]] = []
    phase_class_means: defaultdict[str, list[float]] = defaultdict(list)
    phase_class_magnitudes: defaultdict[str, list[float]] = defaultdict(list)
    for token_id in range(tracker.vocab_size):
        coherence = tracker.get_coherence(token_id)
        grad_magnitude_ema = tracker.get_grad_magnitude_ema(token_id)
        grad_magnitude_mean = tracker.get_grad_magnitude_mean(token_id)
        regime_hint = "mixed"
        if token_id >= tracker.min_token_id and int(tracker.update_count[token_id].item()) >= tracker.min_updates:
            if grad_magnitude_ema <= low_mag_threshold and coherence <= 0.1:
                regime_hint = "starved"
            elif grad_magnitude_ema >= high_mag_threshold and coherence <= 0.1:
                regime_hint = "collided"
            elif grad_magnitude_ema >= high_mag_threshold and coherence >= 0.8:
                regime_hint = "crystallized"
        row = {
            "token_id": token_id,
            "coherence": coherence,
            "grad_magnitude_ema": grad_magnitude_ema,
            "grad_magnitude_mean": grad_magnitude_mean,
            "grad_energy_total": tracker.get_grad_energy_total(token_id),
            "grad_magnitude_max": tracker.get_grad_magnitude_max(token_id),
            "coherence_numerator_l2_ema": float(tracker.ema_grad[token_id].norm().item()),
            "coherence_denominator_l2_ema": float(tracker.ema_grad_norm[token_id].item()),
            "update_count": int(tracker.update_count[token_id].item()),
            "last_update_step": int(tracker.last_update_step[token_id].item()),
            "is_dynamic": token_id >= tracker.min_token_id,
            "is_mature": int(tracker.update_count[token_id].item()) >= tracker.min_updates,
            "regime_hint": regime_hint,
        }
        row["ema_grad_norm"] = row["grad_magnitude_ema"]
        row["ema_vector_norm"] = row["coherence_numerator_l2_ema"]
        if token_id in phase_class_by_token:
            row["phase_classification"] = phase_class_by_token[token_id]
            if row["is_mature"]:
                phase_class_means[row["phase_classification"]].append(coherence)
                phase_class_magnitudes[row["phase_classification"]].append(grad_magnitude_ema)
        row.update(vocab_metadata.get(token_id, {}))
        rows.append(row)

    return {
        "summary": summary,
        "phase_class_means": {
            phase: {
                "n": len(values),
                "mean_coherence": float(sum(values) / len(values)) if values else 0.0,
                "mean_grad_magnitude_ema": float(sum(phase_class_magnitudes[phase]) / len(phase_class_magnitudes[phase]))
                if phase_class_magnitudes[phase]
                else 0.0,
            }
            for phase, values in sorted(phase_class_means.items())
        },
        "rows": rows,
    }


def write_coherence_snapshot(
    path: str | Path,
    tracker: GradientCoherenceTracker,
    *,
    vocabulary_json_path: str | Path | None = None,
    phase_space_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = build_coherence_snapshot(
        tracker,
        vocabulary_json_path=vocabulary_json_path,
        phase_space_snapshot=phase_space_snapshot,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot
