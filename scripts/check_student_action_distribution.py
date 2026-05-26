import argparse
import dataclasses
import json
import math
import pathlib
import random
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


DEFAULT_DATASET_DIR = pathlib.Path("results/libero_dataset_500")
DEFAULT_MODEL_PATH = pathlib.Path("results/libero_student_bc/best_model.pt")
DEFAULT_OUTPUT_PATH = pathlib.Path("results/libero_student_bc/action_distribution_check.json")
IMAGE_FILENAMES = ("images.npy", "wrist_images.npy", "states.npy", "actions.npy")


@dataclasses.dataclass(frozen=True)
class SampleRef:
    episode_dir: pathlib.Path
    step_idx: int


class LiberoStepDataset(Dataset):
    def __init__(self, dataset_dir: pathlib.Path, *, cache_size: int = 4) -> None:
        self.dataset_dir = dataset_dir
        self.cache_size = cache_size
        self.episode_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
        if not self.episode_dirs:
            raise RuntimeError(f"No episode directories found in {dataset_dir}")

        self.samples: list[SampleRef] = []
        for episode_dir in self.episode_dirs:
            meta_path = episode_dir / "meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing meta.json: {meta_path}")
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            episode_len = int(meta["episode_len"])

            for filename in IMAGE_FILENAMES:
                array_path = episode_dir / filename
                if not array_path.exists():
                    raise FileNotFoundError(f"Missing array file: {array_path}")

            self.samples.extend(SampleRef(episode_dir=episode_dir, step_idx=i) for i in range(episode_len))

        self._cache: "OrderedDict[str, dict[str, np.ndarray]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_episode(self, episode_dir: pathlib.Path) -> dict[str, np.ndarray]:
        key = episode_dir.name
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        arrays = {
            filename: np.load(episode_dir / filename, mmap_mode="r", allow_pickle=False)
            for filename in IMAGE_FILENAMES
        }
        self._cache[key] = arrays
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        sample = self.samples[index]
        arrays = self._load_episode(sample.episode_dir)
        step_idx = sample.step_idx
        return {
            "image": np.asarray(arrays["images.npy"][step_idx]),
            "wrist_image": np.asarray(arrays["wrist_images.npy"][step_idx]),
            "state": np.asarray(arrays["states.npy"][step_idx], dtype=np.float32),
            "action": np.asarray(arrays["actions.npy"][step_idx], dtype=np.float32),
        }


class SmallCNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentBCPolicy(nn.Module):
    def __init__(self, action_dim: int = 7) -> None:
        super().__init__()
        self.image_encoder = SmallCNNEncoder(out_dim=64)
        self.wrist_encoder = SmallCNNEncoder(out_dim=64)
        self.state_mlp = nn.Sequential(
            nn.Linear(8, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 32, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def _normalize_image(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32).permute(0, 3, 1, 2).div(255.0)
        return (x - self.image_mean) / self.image_std

    def forward(self, image: torch.Tensor, wrist_image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(self._normalize_image(image))
        wrist_feat = self.wrist_encoder(self._normalize_image(wrist_image))
        state_feat = self.state_mlp(state.to(dtype=torch.float32))
        fused = torch.cat([image_feat, wrist_feat, state_feat], dim=-1)
        return self.head(fused)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


def _load_model(model_path: pathlib.Path, device: torch.device) -> StudentBCPolicy:
    checkpoint = torch.load(model_path, map_location=device)
    model = StudentBCPolicy(action_dim=7).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _sample_index_subset(dataset_size: int, sample_count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if sample_count <= dataset_size:
        return rng.sample(range(dataset_size), sample_count)
    indices = list(range(dataset_size))
    while len(indices) < sample_count:
        indices.append(rng.randrange(dataset_size))
    return indices


def _collate_batch(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    images = torch.from_numpy(np.stack([item["image"] for item in batch], axis=0))
    wrist_images = torch.from_numpy(np.stack([item["wrist_image"] for item in batch], axis=0))
    states = torch.from_numpy(np.stack([item["state"] for item in batch], axis=0))
    actions = torch.from_numpy(np.stack([item["action"] for item in batch], axis=0))
    return {
        "image": images,
        "wrist_image": wrist_images,
        "state": states,
        "action": actions,
    }


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "has_nan": bool(np.isnan(array).any()) if np.issubdtype(array.dtype, np.inexact) else False,
        "has_inf": bool(np.isinf(array).any()) if np.issubdtype(array.dtype, np.inexact) else False,
    }


def check_student_action_distribution(
    dataset_dir: pathlib.Path,
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    sample_count: int,
    seed: int,
) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    _set_seed(seed)

    dataset = LiberoStepDataset(dataset_dir)
    dataset_size = len(dataset)
    indices = _sample_index_subset(dataset_size, sample_count, seed)
    sampled_dataset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(sampled_dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=_collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_path, device)

    gt_actions: list[np.ndarray] = []
    pred_actions: list[np.ndarray] = []

    for batch in loader:
        image = batch["image"].to(device)
        wrist_image = batch["wrist_image"].to(device)
        state = batch["state"].to(device)
        action = batch["action"].to(device)

        with torch.no_grad():
            pred = model(image, wrist_image, state)

        gt_actions.append(action.cpu().numpy())
        pred_actions.append(pred.cpu().numpy())

    gt = np.concatenate(gt_actions, axis=0)
    pred = np.concatenate(pred_actions, axis=0)

    if gt.shape[0] != sample_count:
        raise RuntimeError(f"Expected {sample_count} samples, got {gt.shape[0]}")

    diff = pred - gt
    overall_mse = float(np.mean(diff**2))
    per_dim_mse = np.mean(diff**2, axis=0).tolist()

    gt_stats = _array_stats(gt)
    pred_stats = _array_stats(pred)
    pred_has_nan = bool(np.isnan(pred).any())
    pred_has_inf = bool(np.isinf(pred).any())

    print(f"Dataset size: {dataset_size}")
    print(f"Sampled steps: {sample_count}")
    print("Ground truth action stats:")
    print(f"  mean={np.array2string(np.asarray(gt_stats['mean']), precision=6, separator=', ')}")
    print(f"  std={np.array2string(np.asarray(gt_stats['std']), precision=6, separator=', ')}")
    print(f"  min={np.array2string(np.asarray(gt_stats['min']), precision=6, separator=', ')}")
    print(f"  max={np.array2string(np.asarray(gt_stats['max']), precision=6, separator=', ')}")
    print("Predicted action stats:")
    print(f"  mean={np.array2string(np.asarray(pred_stats['mean']), precision=6, separator=', ')}")
    print(f"  std={np.array2string(np.asarray(pred_stats['std']), precision=6, separator=', ')}")
    print(f"  min={np.array2string(np.asarray(pred_stats['min']), precision=6, separator=', ')}")
    print(f"  max={np.array2string(np.asarray(pred_stats['max']), precision=6, separator=', ')}")
    print(f"Overall MSE: {overall_mse:.8f}")
    print(f"Per-dim MSE: {np.array2string(np.asarray(per_dim_mse), precision=8, separator=', ')}")
    print(f"Pred has NaN: {pred_has_nan}")
    print(f"Pred has Inf: {pred_has_inf}")

    print("First 10 gt/pred action pairs:")
    first_n = min(10, gt.shape[0])
    for i in range(first_n):
        print(
            f"  [{i}] gt={np.array2string(gt[i], precision=6, separator=', ')} "
            f"pred={np.array2string(pred[i], precision=6, separator=', ')}"
        )

    result = {
        "dataset_dir": str(dataset_dir),
        "model_path": str(model_path),
        "dataset_size": dataset_size,
        "sample_count": sample_count,
        "seed": seed,
        "sampled_indices": indices,
        "gt_action_stats": gt_stats,
        "pred_action_stats": pred_stats,
        "overall_mse": overall_mse,
        "per_dim_mse": per_dim_mse,
        "pred_has_nan": pred_has_nan,
        "pred_has_inf": pred_has_inf,
        "first_10_pairs": [
            {
                "index": i,
                "gt": gt[i].tolist(),
                "pred": pred[i].tolist(),
            }
            for i in range(first_n)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(result), f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Saved result: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the student action distribution against teacher actions.")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    check_student_action_distribution(
        dataset_dir=pathlib.Path(args.dataset_dir),
        model_path=pathlib.Path(args.model_path),
        output_path=pathlib.Path(args.output_path),
        sample_count=args.sample_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
