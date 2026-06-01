"""Offline action adapter for LIBERO action bridge analysis.

This module is intentionally self-contained and depends only on NumPy.
It provides a small safety middle layer for already-recorded actions.npy
files without requiring a live LIBERO environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_VALID_MODES = {"identity", "scale", "clip_only", "smooth_only", "clip_and_smooth"}


def _as_int_list(values: list[int] | tuple[int, ...] | np.ndarray | None) -> list[int] | None:
    if values is None:
        return None
    return [int(value) for value in values]


def _broadcast_bounds(bounds: np.ndarray | float | int | None, action_dim: int) -> np.ndarray | None:
    if bounds is None:
        return None
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(action_dim, float(arr), dtype=np.float64)
    if arr.shape != (action_dim,):
        raise ValueError(
            f"action_low/action_high must be scalar or shape=({action_dim},), got shape={arr.shape}"
        )
    return arr.astype(np.float64, copy=False)


@dataclass
class _ResolvedDims:
    action_dim: int
    motion_dims: list[int]
    gripper_dims: list[int]


class ActionAdapter:
    """Apply offline motion safety transforms to a single action stream."""

    def __init__(
        self,
        mode: str = "identity",
        action_low: np.ndarray | float | int | None = None,
        action_high: np.ndarray | float | int | None = None,
        max_norm: float | None = None,
        max_jump: float | None = None,
        smooth_alpha: float = 1.0,
        scale: float = 1.0,
        motion_dims: list[int] | tuple[int, ...] | np.ndarray | None = None,
        gripper_dims: list[int] | tuple[int, ...] | np.ndarray | None = None,
        threshold_gripper: bool = False,
        gripper_threshold: float = 0.0,
    ) -> None:
        mode = mode.lower().strip()
        if mode not in _VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode}. Expected one of {sorted(_VALID_MODES)}")

        self.mode = mode
        self.action_low = action_low
        self.action_high = action_high
        self.max_norm = max_norm
        self.max_jump = max_jump
        self.smooth_alpha = float(smooth_alpha)
        self.scale = float(scale)
        self.motion_dims = _as_int_list(motion_dims)
        self.gripper_dims = _as_int_list(gripper_dims)
        self.threshold_gripper = bool(threshold_gripper)
        self.gripper_threshold = float(gripper_threshold)

        if not 0.0 <= self.smooth_alpha <= 1.0:
            raise ValueError(f"smooth_alpha must be in [0, 1], got {self.smooth_alpha}")
        if self.max_norm is not None and float(self.max_norm) <= 0.0:
            raise ValueError(f"max_norm must be positive when provided, got {self.max_norm}")
        if self.max_jump is not None and float(self.max_jump) <= 0.0:
            raise ValueError(f"max_jump must be positive when provided, got {self.max_jump}")

        self.reset()

    def reset(self) -> None:
        """Clear the previous-step state."""

        self._prev_raw_motion: np.ndarray | None = None
        self._prev_adapted_motion: np.ndarray | None = None
        self._resolved: _ResolvedDims | None = None

    def resolve_dims(self, action_dim: int) -> tuple[list[int], list[int]]:
        """Resolve motion and gripper indices for a concrete action dimension."""

        resolved = self._resolve_dims(action_dim)
        return resolved.motion_dims, resolved.gripper_dims

    def _resolve_dims(self, action_dim: int) -> _ResolvedDims:
        if self._resolved is not None and self._resolved.action_dim == action_dim:
            return self._resolved

        if self.motion_dims is None:
            if action_dim >= 7:
                motion_dims = list(range(6))
            else:
                motion_dims = list(range(action_dim))
        else:
            motion_dims = list(dict.fromkeys(self.motion_dims))

        if self.gripper_dims is None:
            gripper_dims = [6] if action_dim >= 7 else []
        else:
            gripper_dims = list(dict.fromkeys(self.gripper_dims))

        all_dims = motion_dims + gripper_dims
        if len(set(all_dims)) != len(all_dims):
            raise ValueError(
                f"motion_dims and gripper_dims overlap for action_dim={action_dim}: "
                f"motion_dims={motion_dims}, gripper_dims={gripper_dims}"
            )
        for dim in all_dims:
            if dim < 0 or dim >= action_dim:
                raise ValueError(
                    f"Action dimension index out of range for action_dim={action_dim}: {dim}"
                )

        self._resolved = _ResolvedDims(
            action_dim=action_dim,
            motion_dims=motion_dims,
            gripper_dims=gripper_dims,
        )
        return self._resolved

    def _mode_flags(self) -> dict[str, bool]:
        if self.mode == "identity":
            return {
                "scale": False,
                "clamp": False,
                "norm": False,
                "smooth": False,
                "jump": False,
                "gripper": False,
            }
        if self.mode == "scale":
            return {
                "scale": True,
                "clamp": False,
                "norm": False,
                "smooth": False,
                "jump": False,
                "gripper": True,
            }
        if self.mode == "clip_only":
            return {
                "scale": False,
                "clamp": True,
                "norm": True,
                "smooth": False,
                "jump": True,
                "gripper": True,
            }
        if self.mode == "smooth_only":
            return {
                "scale": False,
                "clamp": False,
                "norm": False,
                "smooth": True,
                "jump": False,
                "gripper": True,
            }
        return {
            "scale": False,
            "clamp": True,
            "norm": True,
            "smooth": True,
            "jump": True,
            "gripper": True,
        }

    @staticmethod
    def _clamp(values: np.ndarray, low: np.ndarray | None, high: np.ndarray | None) -> tuple[np.ndarray, bool]:
        if low is None and high is None:
            return values, False

        clipped = values.copy()
        before = clipped.copy()
        if low is not None:
            clipped = np.maximum(clipped, low)
        if high is not None:
            clipped = np.minimum(clipped, high)
        return clipped, not np.allclose(before, clipped)

    @staticmethod
    def _clip_l2_norm(values: np.ndarray, max_norm: float) -> tuple[np.ndarray, bool]:
        norm = float(np.linalg.norm(values))
        if norm <= max_norm or norm == 0.0:
            return values, False
        return values * (max_norm / norm), True

    @staticmethod
    def _clip_jump(values: np.ndarray, prev_values: np.ndarray, max_jump: float) -> tuple[np.ndarray, bool]:
        diff = values - prev_values
        jump = float(np.linalg.norm(diff))
        if jump <= max_jump or jump == 0.0:
            return values, False
        return prev_values + diff * (max_jump / jump), True

    def adapt(self, action: np.ndarray | list[float] | tuple[float, ...]) -> tuple[np.ndarray, dict[str, Any]]:
        """Adapt one action frame and return (adapted_action, info)."""

        raw_action = np.asarray(action, dtype=np.float64)
        if raw_action.ndim != 1:
            raise ValueError(f"adapt() expects a 1D action, got shape={raw_action.shape}")

        action_dim = int(raw_action.shape[0])
        resolved = self._resolve_dims(action_dim)
        flags = self._mode_flags()

        raw_motion = raw_action[resolved.motion_dims] if resolved.motion_dims else np.zeros(0, dtype=np.float64)
        raw_gripper = raw_action[resolved.gripper_dims] if resolved.gripper_dims else np.zeros(0, dtype=np.float64)

        adapted_action = raw_action.copy()
        adapted_motion = raw_motion.copy()
        adapted_gripper = raw_gripper.copy()

        info: dict[str, Any] = {
            "clipped_by_bound": False,
            "clipped_by_norm": False,
            "clipped_by_jump": False,
            "smoothed": False,
            "scaled": False,
            "gripper_thresholded": False,
            "raw_norm_motion": float(np.linalg.norm(raw_motion)) if raw_motion.size else 0.0,
            "adapted_norm_motion": 0.0,
            "action_jump_before_motion": None,
            "action_jump_after_motion": None,
        }

        if self._prev_raw_motion is not None and raw_motion.size:
            info["action_jump_before_motion"] = float(np.linalg.norm(raw_motion - self._prev_raw_motion))

        if self.mode == "identity":
            adapted_action[resolved.motion_dims] = adapted_motion
            adapted_action[resolved.gripper_dims] = adapted_gripper
            info["adapted_norm_motion"] = float(np.linalg.norm(adapted_motion)) if adapted_motion.size else 0.0
            if self._prev_adapted_motion is not None and adapted_motion.size:
                info["action_jump_after_motion"] = float(
                    np.linalg.norm(adapted_motion - self._prev_adapted_motion)
                )
            self._prev_raw_motion = raw_motion.copy()
            self._prev_adapted_motion = adapted_motion.copy()
            return adapted_action.astype(np.float32, copy=False), info

        if flags["scale"] and self.scale != 1.0 and adapted_motion.size:
            adapted_motion = adapted_motion * self.scale
            info["scaled"] = True

        if flags["smooth"] and adapted_motion.size and self._prev_adapted_motion is not None and self.smooth_alpha != 1.0:
            adapted_motion = self.smooth_alpha * adapted_motion + (1.0 - self.smooth_alpha) * self._prev_adapted_motion
            info["smoothed"] = True

        if flags["norm"] and self.max_norm is not None and adapted_motion.size:
            adapted_motion, clipped = self._clip_l2_norm(adapted_motion, float(self.max_norm))
            info["clipped_by_norm"] = clipped

        if flags["jump"] and self.max_jump is not None and adapted_motion.size and self._prev_adapted_motion is not None:
            adapted_motion, clipped = self._clip_jump(adapted_motion, self._prev_adapted_motion, float(self.max_jump))
            info["clipped_by_jump"] = clipped

        adapted_action[resolved.motion_dims] = adapted_motion

        if flags["gripper"] and adapted_gripper.size:
            gripper_low = np.full(adapted_gripper.shape, -1.0, dtype=np.float64)
            gripper_high = np.full(adapted_gripper.shape, 1.0, dtype=np.float64)
            adapted_gripper, bound_clipped = self._clamp(adapted_gripper, gripper_low, gripper_high)
            info["clipped_by_bound"] = info["clipped_by_bound"] or bound_clipped

            if self.threshold_gripper:
                adapted_gripper = np.where(adapted_gripper >= self.gripper_threshold, 1.0, -1.0)
                info["gripper_thresholded"] = True

        adapted_action[resolved.gripper_dims] = adapted_gripper

        if flags["clamp"]:
            low = _broadcast_bounds(self.action_low, action_dim)
            high = _broadcast_bounds(self.action_high, action_dim)
            if low is not None or high is not None:
                before = adapted_action.copy()
                if low is not None:
                    adapted_action = np.maximum(adapted_action, low)
                if high is not None:
                    adapted_action = np.minimum(adapted_action, high)
                info["clipped_by_bound"] = info["clipped_by_bound"] or (not np.allclose(before, adapted_action))

        adapted_motion = adapted_action[resolved.motion_dims] if resolved.motion_dims else np.zeros(0, dtype=np.float64)
        info["adapted_norm_motion"] = float(np.linalg.norm(adapted_motion)) if adapted_motion.size else 0.0
        if self._prev_adapted_motion is not None and adapted_motion.size:
            info["action_jump_after_motion"] = float(np.linalg.norm(adapted_motion - self._prev_adapted_motion))

        self._prev_raw_motion = raw_motion.copy()
        self._prev_adapted_motion = adapted_motion.copy()
        return adapted_action.astype(np.float32, copy=False), info

    def adapt_sequence(
        self, actions: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...]
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Adapt a full [T, action_dim] action sequence step by step."""

        sequence = np.asarray(actions, dtype=np.float64)
        if sequence.ndim != 2:
            raise ValueError(f"adapt_sequence() expects a 2D array, got shape={sequence.shape}")

        self.reset()
        adapted_steps: list[np.ndarray] = []
        infos: list[dict[str, Any]] = []
        for step in sequence:
            adapted_step, info = self.adapt(step)
            adapted_steps.append(adapted_step)
            infos.append(info)
        return np.stack(adapted_steps, axis=0), infos
