from __future__ import annotations

from dataclasses import dataclass

from telemetry import TelemetryBuffer
from velocity import (
    FIRST_DYNAMIC_TOKEN_ID,
    VelocityTracker,
    calibrate_velocity_thresholds,
    classify_token as classify_velocity_token,
    is_quiescent_classification,
)


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    if mean_value == 0.0:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return (variance ** 0.5) / mean_value


@dataclass
class MutationConfig:
    M: int = 500
    K: int = 5

    V_EVERY: int = 50
    panic_threshold: float | None = None
    work_threshold: float | None = None
    work_floor: float | None = None
    panic_quantile: float = 0.75
    work_quantile: float = 0.75
    work_floor_quantile: float = 0.25
    min_observations: int = 500

    fusion_threshold: float = 1.5
    min_fusion_frequency: int = 50
    noise_cv_threshold: float = 0.8

    static_core_size: int = 150
    min_token_age: int = 2
    cooldown_cycles: int = 3

    surgeon_lr_multiplier: float = 3.0
    surgeon_warmup_steps: int = 50
    ema_alpha: float = 0.3


class MutationEngine:
    """
    Phase-3 analysis-side mutation logic for DGT-v1.

    This module does not yet perform trie swaps or matrix surgery. It turns the new
    telemetry surfaces into actionable candidate lists so Phase 4 can inspect and
    validate them before we close the loop.
    """

    def __init__(
        self,
        telemetry: TelemetryBuffer,
        velocity_tracker: VelocityTracker,
        config: MutationConfig | None = None,
        static_core_ids: set[int] | None = None,
        token_birth_cycle: dict[int, int] | None = None,
        cycle_count: int = 0,
    ):
        self.telemetry = telemetry
        self.velocity_tracker = velocity_tracker
        self.config = config or MutationConfig()
        self.static_core_ids = static_core_ids or set()
        self.token_birth_cycle = token_birth_cycle or {}
        self.cycle_count = cycle_count

    def _thresholds(self) -> dict[str, float | int]:
        thresholds = calibrate_velocity_thresholds(
            self.velocity_tracker,
            static_core_ids=self.static_core_ids,
            min_observations=self.config.min_observations,
            panic_quantile=self.config.panic_quantile,
            work_quantile=self.config.work_quantile,
            work_floor_quantile=self.config.work_floor_quantile,
        )
        if self.config.panic_threshold is not None:
            thresholds["panic_threshold"] = self.config.panic_threshold
        if self.config.work_threshold is not None:
            thresholds["work_threshold"] = self.config.work_threshold
        if self.config.work_floor is not None:
            thresholds["work_floor"] = self.config.work_floor
        return thresholds

    def classify_token(self, token_id: int) -> str:
        thresholds = self._thresholds()
        return classify_velocity_token(
            self.velocity_tracker,
            token_id,
            static_core_ids=self.static_core_ids,
            min_observations=self.config.min_observations,
            panic_threshold=float(thresholds["panic_threshold"]),
            work_threshold=float(thresholds["work_threshold"]),
            work_floor=float(thresholds["work_floor"]),
        )

    def get_fission_candidates(self) -> list[dict[str, float | int | str]]:
        thresholds = self._thresholds()
        debris: list[dict[str, float | int | str]] = []
        quiescent: list[dict[str, float | int | str]] = []

        for token_id in range(FIRST_DYNAMIC_TOKEN_ID, self.velocity_tracker.vocab_size):
            if token_id in self.static_core_ids:
                continue

            classification = classify_velocity_token(
                self.velocity_tracker,
                token_id,
                static_core_ids=self.static_core_ids,
                min_observations=self.config.min_observations,
                panic_threshold=float(thresholds["panic_threshold"]),
                work_threshold=float(thresholds["work_threshold"]),
                work_floor=float(thresholds["work_floor"]),
            )
            if classification != "DEBRIS" and not is_quiescent_classification(classification):
                continue

            age_cycles = self.cycle_count - self.token_birth_cycle.get(token_id, 0)
            if age_cycles < self.config.min_token_age:
                continue

            candidate = {
                "token_id": token_id,
                "classification": classification,
                "panic_ratio": float(self.velocity_tracker.panic_ratio[token_id].item()),
                "active_work": float(self.velocity_tracker.active_work[token_id].item()),
                "observation_count": int(self.velocity_tracker.observation_count[token_id].item()),
                "age_cycles": int(age_cycles),
            }
            if classification == "DEBRIS":
                debris.append(candidate)
            else:
                quiescent.append(candidate)

        debris.sort(key=lambda row: (row["panic_ratio"], row["active_work"]), reverse=True)
        quiescent.sort(key=lambda row: (row["active_work"], row["panic_ratio"]))
        return (debris + quiescent)[: self.config.K]

    def get_fusion_candidates(self) -> list[dict[str, float | int | str]]:
        byte_run_losses = self.telemetry.byte_run_losses
        if not byte_run_losses:
            return []

        all_losses = [loss for losses in byte_run_losses.values() for loss in losses]
        if not all_losses:
            return []
        corpus_mean = sum(all_losses) / len(all_losses)
        loss_cut = corpus_mean * self.config.fusion_threshold

        candidates: list[dict[str, float | int | str]] = []
        for byte_string, losses in byte_run_losses.items():
            if len(byte_string) < 2 or len(byte_string) > 12:
                continue
            if len(losses) < self.config.min_fusion_frequency:
                continue

            mean_loss = sum(losses) / len(losses)
            if mean_loss <= loss_cut:
                continue

            cv = _coefficient_of_variation(losses)
            if cv > self.config.noise_cv_threshold:
                continue

            candidates.append(
                {
                    "byte_string_hex": byte_string.hex(),
                    "string": byte_string.decode("utf-8", errors="replace"),
                    "mean_loss": float(mean_loss),
                    "mean_panic_ratio": float(self.velocity_tracker.get_byte_run_panic(byte_string)),
                    "frequency": int(len(losses)),
                    "coefficient_of_variation": float(cv),
                }
            )

        candidates.sort(key=lambda row: (row["mean_loss"], row["mean_panic_ratio"]), reverse=True)
        return candidates[: self.config.K]
