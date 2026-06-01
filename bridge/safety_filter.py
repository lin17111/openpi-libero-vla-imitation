"""Offline safety filter for action sequences.

This module is intentionally independent from LIBERO / OpenPI runtime code.
It provides a lightweight rule-based safety layer that can be applied after
action adaptation, before any environment step or robot controller command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _to_1d_array(value: Any, action_dim: int, *, allow_none: bool = False) -> np.ndarray | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError("Expected a scalar or array-like value, got None.")

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return np.full(action_dim, float(arr), dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] == action_dim:
            return arr.astype(np.float32, copy=False)
        if arr.shape[0] == 1:
            return np.full(action_dim, float(arr[0]), dtype=np.float32)
        raise ValueError(f"Expected a scalar or length-{action_dim} array, got shape {arr.shape}.")
    raise ValueError(f"Expected a scalar or 1D array, got shape {arr.shape}.")


def _as_index_list(value: list[int] | tuple[int, ...] | np.ndarray | None) -> list[int] | None:
    if value is None:
        return None
    return [int(x) for x in value]


@dataclass
class _ResolvedConfig:
    action_dim: int
    motion_dims: list[int]
    gripper_dims: list[int]
    action_low: np.ndarray | None
    action_high: np.ndarray | None
    workspace_low: np.ndarray | None
    workspace_high: np.ndarray | None


class SafetyFilter:
    """Rule-based offline safety filter for action vectors.

    The default configuration treats the first six dimensions as motion and the
    seventh dimension as gripper when action_dim >= 7. For smaller action
    vectors, all dimensions are treated as motion.
    """

    def __init__(
        self,
        action_low=None,
        action_high=None,
        max_norm=None,
        max_jump=None,
        max_consecutive_triggers: int = 5,
        reject_on_violation: bool = False,
        fallback_to_previous: bool = True,
        motion_dims=None,
        gripper_dims=None,
        workspace_low=None,
        workspace_high=None,
    ) -> None:
        self._raw_action_low = action_low
        self._raw_action_high = action_high
        self.max_norm = 1.5 if max_norm is None else float(max_norm)
        self.max_jump = 0.5 if max_jump is None else float(max_jump)
        self.max_consecutive_triggers = int(max_consecutive_triggers)
        self.reject_on_violation = bool(reject_on_violation)
        self.fallback_to_previous = bool(fallback_to_previous)
        self._raw_motion_dims = _as_index_list(motion_dims)
        self._raw_gripper_dims = _as_index_list(gripper_dims)
        self._raw_workspace_low = workspace_low
        self._raw_workspace_high = workspace_high

        self.previous_action: np.ndarray | None = None
        self.previous_safe_action: np.ndarray | None = None
        self.consecutive_trigger_count = 0

        self._resolved: _ResolvedConfig | None = None

    def reset(self) -> None:
        """Reset filter state between sequences."""

        self.previous_action = None
        self.previous_safe_action = None
        self.consecutive_trigger_count = 0
        self._resolved = None

    def _resolve_config(self, action_dim: int) -> _ResolvedConfig:
        if self._resolved is not None and self._resolved.action_dim != action_dim:
            raise ValueError(
                f"Action dimension changed from {self._resolved.action_dim} to {action_dim}. "
                "Call reset() before filtering a sequence with a different action dimension."
            )

        if self._resolved is not None:
            return self._resolved

        if self._raw_motion_dims is None:
            motion_dims = list(range(6)) if action_dim >= 7 else list(range(action_dim))
        else:
            motion_dims = list(self._raw_motion_dims)

        if self._raw_gripper_dims is None:
            gripper_dims = [6] if action_dim >= 7 else []
        else:
            gripper_dims = list(self._raw_gripper_dims)

        all_dims = set(range(action_dim))
        for name, dims in (("motion_dims", motion_dims), ("gripper_dims", gripper_dims)):
            for dim in dims:
                if dim not in all_dims:
                    raise ValueError(f"{name} contains out-of-range dimension {dim} for action_dim={action_dim}.")

        if set(motion_dims) & set(gripper_dims):
            raise ValueError("motion_dims and gripper_dims must not overlap.")

        action_low = _to_1d_array(self._raw_action_low, action_dim, allow_none=True)
        action_high = _to_1d_array(self._raw_action_high, action_dim, allow_none=True)
        workspace_low = _to_1d_array(self._raw_workspace_low, 3, allow_none=True)
        workspace_high = _to_1d_array(self._raw_workspace_high, 3, allow_none=True)

        self._resolved = _ResolvedConfig(
            action_dim=action_dim,
            motion_dims=motion_dims,
            gripper_dims=gripper_dims,
            action_low=action_low,
            action_high=action_high,
            workspace_low=workspace_low,
            workspace_high=workspace_high,
        )
        return self._resolved

    @staticmethod
    def _compute_motion_norm(action: np.ndarray, motion_dims: list[int]) -> float:
        if not motion_dims:
            return 0.0
        return float(np.linalg.norm(action[motion_dims].astype(np.float64)))

    @staticmethod
    def _compute_motion_jump(current_motion: np.ndarray, previous_motion: np.ndarray | None) -> float | None:
        if previous_motion is None:
            return None
        return float(np.linalg.norm((current_motion - previous_motion).astype(np.float64)))

    def _apply_bounds(self, action: np.ndarray, cfg: _ResolvedConfig) -> tuple[np.ndarray, bool]:
        bounded = action.copy()
        violation = False

        if cfg.action_low is not None:
            new_bounded = np.maximum(bounded, cfg.action_low)
            violation = violation or bool(np.any(new_bounded != bounded))
            bounded = new_bounded
        if cfg.action_high is not None:
            new_bounded = np.minimum(bounded, cfg.action_high)
            violation = violation or bool(np.any(new_bounded != bounded))
            bounded = new_bounded

        if cfg.gripper_dims:
            gripper = bounded[cfg.gripper_dims]
            clipped = np.clip(gripper, -1.0, 1.0)
            violation = violation or bool(np.any(clipped != gripper))
            bounded[cfg.gripper_dims] = clipped

        return bounded, violation

    def _apply_motion_norm(self, action: np.ndarray, cfg: _ResolvedConfig) -> tuple[np.ndarray, bool, float, float]:
        if not cfg.motion_dims:
            raw_norm = 0.0
            return action, False, raw_norm, raw_norm

        current = action.copy()
        motion = current[cfg.motion_dims]
        raw_norm = float(np.linalg.norm(motion.astype(np.float64)))
        violation = bool(self.max_norm is not None and raw_norm > self.max_norm)
        if violation and raw_norm > 0.0:
            motion = motion * (self.max_norm / raw_norm)
            current[cfg.motion_dims] = motion
        safe_norm = float(np.linalg.norm(current[cfg.motion_dims].astype(np.float64)))
        return current, violation, raw_norm, safe_norm

    def _apply_motion_jump(
        self,
        action: np.ndarray,
        cfg: _ResolvedConfig,
        previous_safe_action: np.ndarray | None,
    ) -> tuple[np.ndarray, bool, float | None, float | None]:
        if previous_safe_action is None or not cfg.motion_dims:
            return action, False, None, None

        current = action.copy()
        previous_motion = previous_safe_action[cfg.motion_dims]
        current_motion = current[cfg.motion_dims]
        jump_before = float(np.linalg.norm((current_motion - previous_motion).astype(np.float64)))
        violation = bool(self.max_jump is not None and jump_before > self.max_jump)

        if violation and jump_before > 0.0:
            direction = current_motion - previous_motion
            current_motion = previous_motion + direction * (self.max_jump / jump_before)
            current[cfg.motion_dims] = current_motion

        jump_after = float(np.linalg.norm((current[cfg.motion_dims] - previous_motion).astype(np.float64)))
        return current, violation, jump_before, jump_after

    def _apply_workspace_check(self, ee_pose: np.ndarray | None, cfg: _ResolvedConfig) -> bool:
        if ee_pose is None or cfg.workspace_low is None or cfg.workspace_high is None:
            return False
        pose = np.asarray(ee_pose, dtype=np.float32).reshape(-1)
        if pose.shape[0] < 3:
            raise ValueError(f"ee_pose must have at least 3 values, got shape {pose.shape}.")
        xyz = pose[:3]
        return bool(np.any(xyz < cfg.workspace_low) or np.any(xyz > cfg.workspace_high))

    def filter(self, action, ee_pose=None):
        """Filter a single action and return the safe action plus diagnostics."""

        raw_action = np.asarray(action, dtype=np.float32)
        if raw_action.ndim != 1:
            raise ValueError(f"Expected a 1D action, got shape {raw_action.shape}.")

        cfg = self._resolve_config(int(raw_action.shape[0]))
        previous_safe_action = None if self.previous_safe_action is None else self.previous_safe_action.copy()

        safe_action = raw_action.copy()
        trigger_reasons: list[str] = []
        bound_violation = False
        norm_violation = False
        jump_violation = False
        workspace_violation = False

        safe_action, bound_violation = self._apply_bounds(safe_action, cfg)
        if bound_violation:
            trigger_reasons.append("bound")

        safe_action, norm_violation, raw_motion_norm, safe_motion_norm = self._apply_motion_norm(safe_action, cfg)
        if norm_violation:
            trigger_reasons.append("norm")

        safe_action, jump_violation, action_jump_before, action_jump_after = self._apply_motion_jump(
            safe_action, cfg, previous_safe_action
        )
        if jump_violation:
            trigger_reasons.append("jump")

        workspace_violation = self._apply_workspace_check(ee_pose, cfg)
        if workspace_violation:
            trigger_reasons.append("workspace")

        safety_triggered = bool(bound_violation or norm_violation or jump_violation or workspace_violation)
        abnormal_sequence = False
        reject_action = False

        if safety_triggered:
            self.consecutive_trigger_count += 1
        else:
            self.consecutive_trigger_count = 0

        if self.consecutive_trigger_count >= self.max_consecutive_triggers and self.max_consecutive_triggers > 0:
            abnormal_sequence = True
            trigger_reasons.append("consecutive_triggers")
            safety_triggered = True

        if self.reject_on_violation and safety_triggered:
            reject_action = True
            trigger_reasons.append("reject_on_violation")
            if self.fallback_to_previous and previous_safe_action is not None:
                safe_action = previous_safe_action.copy()
            else:
                safe_action = np.zeros_like(raw_action)

        safe_motion_norm = self._compute_motion_norm(safe_action, cfg.motion_dims)
        if previous_safe_action is None:
            action_jump_after = None
        else:
            action_jump_after = float(
                np.linalg.norm((safe_action[cfg.motion_dims] - previous_safe_action[cfg.motion_dims]).astype(np.float64))
            ) if cfg.motion_dims else 0.0

        info = {
            "safety_triggered": bool(safety_triggered),
            "bound_violation": bool(bound_violation),
            "norm_violation": bool(norm_violation),
            "jump_violation": bool(jump_violation),
            "workspace_violation": bool(workspace_violation),
            "abnormal_sequence": bool(abnormal_sequence),
            "reject_action": bool(reject_action),
            "trigger_reasons": trigger_reasons,
            "raw_motion_norm": float(raw_motion_norm),
            "safe_motion_norm": float(safe_motion_norm),
            "action_jump_before": action_jump_before,
            "action_jump_after": action_jump_after,
            "consecutive_trigger_count": int(self.consecutive_trigger_count),
        }

        self.previous_action = raw_action.copy()
        self.previous_safe_action = safe_action.copy()
        return safe_action.astype(np.float32, copy=False), info

    def filter_sequence(self, actions, ee_poses=None):
        """Filter a sequence of actions step-by-step."""

        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.ndim != 2:
            raise ValueError(f"Expected an array shaped [T, action_dim], got shape {action_array.shape}.")

        if ee_poses is not None:
            ee_pose_array = np.asarray(ee_poses, dtype=np.float32)
            if ee_pose_array.ndim != 2:
                raise ValueError(f"Expected ee_poses shaped [T, pose_dim], got shape {ee_pose_array.shape}.")
            if ee_pose_array.shape[0] != action_array.shape[0]:
                raise ValueError(
                    f"actions and ee_poses must have the same length, got {action_array.shape[0]} and {ee_pose_array.shape[0]}."
                )
        else:
            ee_pose_array = None

        safe_actions = []
        infos = []
        for idx, action in enumerate(action_array):
            ee_pose = None if ee_pose_array is None else ee_pose_array[idx]
            safe_action, info = self.filter(action, ee_pose=ee_pose)
            safe_actions.append(safe_action)
            infos.append(info)
        return np.asarray(safe_actions, dtype=np.float32), infos
