import argparse
import math
import os
import pathlib
import random

if "DISPLAY" not in os.environ and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import torch
from torch import nn

from openpi_client import image_tools


DEFAULT_MODEL_PATH = pathlib.Path("results/libero_student_chunk_bc/best_model.pt")
LIBERO_ENV_RESOLUTION = 256
LIBERO_NUM_STEPS_WAIT = 10


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


class StudentChunkBCPolicy(nn.Module):
    def __init__(self, chunk_size: int, action_dim: int = 7) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
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
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, chunk_size * action_dim),
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


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _get_libero_env(task, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _build_student_obs(raw_obs: dict, resize_size: int = 224) -> dict[str, np.ndarray]:
    image = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_image, resize_size, resize_size))
    state = np.asarray(
        np.concatenate(
            (
                raw_obs["robot0_eef_pos"],
                _quat2axisangle(np.asarray(raw_obs["robot0_eef_quat"]).copy()),
                raw_obs["robot0_gripper_qpos"],
            )
        ),
        dtype=np.float32,
    )
    return {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
    }


def _load_student(model_path: pathlib.Path, device: torch.device, chunk_size: int) -> StudentChunkBCPolicy:
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_chunk_size = int(checkpoint.get("model_config", {}).get("chunk_size", chunk_size))
    if checkpoint_chunk_size != chunk_size:
        raise ValueError(
            f"Checkpoint chunk_size={checkpoint_chunk_size} does not match --chunk-size={chunk_size}"
        )
    model = StudentChunkBCPolicy(chunk_size=chunk_size, action_dim=7).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _print_chunk_stats(ep_idx: int, chunk_idx: int, action_chunk: np.ndarray) -> None:
    print(
        f"[episode={ep_idx} chunk={chunk_idx}]",
        f"shape={list(action_chunk.shape)}",
        f"min={action_chunk.min():.6f}",
        f"max={action_chunk.max():.6f}",
        f"mean={action_chunk.mean():.6f}",
        f"std={action_chunk.std():.6f}",
    )


def eval_student_chunk_bc(
    model_path: pathlib.Path,
    task_suite_name: str,
    max_episodes_total: int | None,
    seed: int,
    chunk_size: int,
) -> None:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_student(model_path, device, chunk_size)

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    if task_suite_name == "libero_spatial":
        max_steps = 220
    elif task_suite_name == "libero_object":
        max_steps = 280
    elif task_suite_name == "libero_goal":
        max_steps = 300
    elif task_suite_name == "libero_10":
        max_steps = 520
    elif task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {task_suite_name}")

    total_episodes = 0
    total_successes = 0
    total_episode_lengths = 0
    stop_collection = False

    for task_id in range(num_tasks_in_suite):
        if stop_collection:
            break

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        episodes_this_task = len(initial_states)

        for episode_idx in range(episodes_this_task):
            if max_episodes_total is not None and total_episodes >= max_episodes_total:
                stop_collection = True
                break

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            success = False

            for _ in range(LIBERO_NUM_STEPS_WAIT):
                obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
                t += 1
                if done or bool(env.check_success()):
                    success = True
                    break

            chunk_idx = 0
            while t < max_steps and not success:
                student_obs = _build_student_obs(obs)
                image = torch.from_numpy(student_obs["observation/image"]).unsqueeze(0).to(device)
                wrist_image = torch.from_numpy(student_obs["observation/wrist_image"]).unsqueeze(0).to(device)
                state = torch.from_numpy(student_obs["observation/state"]).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = model(image, wrist_image, state).view(1, chunk_size, 7).squeeze(0).cpu().numpy()

                pred = np.clip(pred, -1.0, 1.0)
                if episode_idx < 2:
                    _print_chunk_stats(episode_idx, chunk_idx, pred)
                chunk_idx += 1

                for step_action in pred:
                    if t >= max_steps or success:
                        break
                    obs, reward, done, info = env.step(step_action.tolist())
                    t += 1
                    if done or bool(env.check_success()):
                        success = True
                        break

            total_episodes += 1
            total_successes += int(success)
            total_episode_lengths += t

            success_rate = total_successes / total_episodes if total_episodes else 0.0
            print(
                f"[task={task_id} episode={episode_idx}] "
                f"success={success} "
                f"episode_len={t} "
                f"total_episodes={total_episodes} "
                f"success_rate={success_rate:.6f}"
            )

    avg_episode_length = total_episode_lengths / total_episodes if total_episodes else 0.0
    success_rate = total_successes / total_episodes if total_episodes else 0.0
    print(f"Total episodes: {total_episodes}")
    print(f"Successes: {total_successes}")
    print(f"Success rate: {success_rate:.6f}")
    print(f"Avg episode length: {avg_episode_length:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a chunk-level LIBERO student policy.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--task-suite-name", type=str, default="libero_spatial")
    parser.add_argument("--max-episodes-total", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=10)
    args = parser.parse_args()

    eval_student_chunk_bc(
        model_path=pathlib.Path(args.model_path),
        task_suite_name=args.task_suite_name,
        max_episodes_total=args.max_episodes_total,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
