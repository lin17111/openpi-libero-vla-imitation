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
from torch.utils.data import Subset


DEFAULT_DATASET_DIR = pathlib.Path("results/libero_dataset_500")
DEFAULT_SAVE_DIR = pathlib.Path("results/libero_student_bc")
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
        self.episode_lengths: dict[str, int] = {}
        for episode_dir in self.episode_dirs:
            meta_path = episode_dir / "meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing meta.json: {meta_path}")
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            episode_len = int(meta["episode_len"])
            self.episode_lengths[episode_dir.name] = episode_len

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


def _prepare_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image = batch["image"].to(device)
    wrist_image = batch["wrist_image"].to(device)
    state = batch["state"].to(device)
    action = batch["action"].to(device)
    return image, wrist_image, state, action


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        image, wrist_image, state, action = _prepare_batch(batch, device)
        batch_size = action.shape[0]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(image, wrist_image, state)
        loss = criterion(pred, action)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return float("nan")
    return total_loss / total_samples


def _print_batch_shapes(batch: dict[str, torch.Tensor]) -> None:
    print("Single batch shapes:")
    print(f"  image: {tuple(batch['image'].shape)}")
    print(f"  wrist_image: {tuple(batch['wrist_image'].shape)}")
    print(f"  state: {tuple(batch['state'].shape)}")
    print(f"  action: {tuple(batch['action'].shape)}")


def train_student_bc(
    dataset_dir: pathlib.Path,
    save_dir: pathlib.Path,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    val_ratio: float,
) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")

    save_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(7)

    dataset = LiberoStepDataset(dataset_dir)
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise RuntimeError(f"Need at least 2 samples, found {dataset_size}")

    val_size = max(1, int(round(dataset_size * val_ratio)))
    if val_size >= dataset_size:
        val_size = dataset_size - 1
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(7)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(f"Dataset samples: {dataset_size}")
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")

    preview_batch = next(iter(train_loader))
    _print_batch_shapes(preview_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentBCPolicy().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_log: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "save_dir": str(save_dir),
        "dataset_size": dataset_size,
        "train_size": train_size,
        "val_size": val_size,
        "batch_size": batch_size,
        "lr": lr,
        "num_workers": num_workers,
        "val_ratio": val_ratio,
        "device": str(device),
        "epochs": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, criterion, device)
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, None, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        }
        train_log["epochs"].append(epoch_record)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        last_path = save_dir / "last_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "model_config": {
                    "action_dim": 7,
                },
            },
            last_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "model_config": {
                        "action_dim": 7,
                    },
                },
                save_dir / "best_model.pt",
            )

        train_log["best_val_loss"] = best_val_loss
        train_log["best_epoch"] = best_epoch
        with (save_dir / "train_log.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(train_log), f, indent=2, ensure_ascii=False)
            f.write("\n")

    print(f"Saved last model to {save_dir / 'last_model.pt'}")
    print(f"Saved best model to {save_dir / 'best_model.pt'}")
    print(f"Saved training log to {save_dir / 'train_log.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal offline BC student on collected LIBERO data.")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    train_student_bc(
        dataset_dir=pathlib.Path(args.dataset_dir),
        save_dir=pathlib.Path(args.save_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
