import collections
import dataclasses
import json
import logging
import math
import pathlib
import shutil

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
DEBUG_ROOT = pathlib.Path("results/libero_debug")
DEBUG_VIDEOS_DIR = DEBUG_ROOT / "videos"


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Data collection parameters
    #################################################################################################################
    save_success_only: bool = False
    max_episodes_total: int | None = None
    dataset_out_dir: str = "results/libero_dataset"

    #################################################################################################################
    # Debug parameters
    #################################################################################################################
    debug_max_steps: int | None = None
    debug_num_episodes: int | None = None

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    DEBUG_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_out_dir = pathlib.Path(args.dataset_out_dir)
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    if args.debug_max_steps is not None:
        max_steps = min(max_steps, args.debug_max_steps)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    saved_episodes = 0
    saved_successes = 0
    saved_steps = 0
    stop_collection = False

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if stop_collection:
            break

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        episodes_this_task = args.num_trials_per_task
        if args.debug_num_episodes is not None:
            episodes_this_task = min(episodes_this_task, args.debug_num_episodes)

        for episode_idx in tqdm.tqdm(range(episodes_this_task)):
            if stop_collection:
                break

            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            replay_images = []
            executed_actions: list[np.ndarray] = []
            episode_images: list[np.ndarray] = []
            episode_wrist_images: list[np.ndarray] = []
            episode_states: list[np.ndarray] = []
            episode_actions: list[np.ndarray] = []
            episode_rewards: list[float] = []
            episode_dones: list[bool] = []
            prompt = str(task_description)
            episode_stats: dict = {
                "task_id": task_id,
                "episode_idx": episode_idx,
                "task_description": task_description,
                "debug_max_steps": args.debug_max_steps,
                "debug_num_episodes": args.debug_num_episodes,
                "chunk_stats": [],
                "step_stats": [],
            }

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    state = np.asarray(
                        np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        dtype=np.float32,
                    )

                    replay_images.append(img)

                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state,
                        "prompt": prompt,
                    }

                    _print_observation_stats(element)

                    if not action_plan:
                        action_chunk = client.infer(element)["actions"]
                        action_chunk_np = np.asarray(action_chunk)
                        chunk_stats = _array_stats(action_chunk_np)
                        episode_stats["chunk_stats"].append(chunk_stats)
                        print(
                            "[action_chunk]",
                            f"shape={chunk_stats['shape']}",
                            f"min={chunk_stats['min']:.6f}",
                            f"max={chunk_stats['max']:.6f}",
                            f"mean={chunk_stats['mean']:.6f}",
                            f"std={chunk_stats['std']:.6f}",
                        )
                        assert (
                            len(action_chunk_np) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk_np)} steps."
                        action_plan.extend(action_chunk_np[: args.replan_steps])

                    action = np.asarray(action_plan.popleft(), dtype=np.float32)
                    step_stats = _array_stats(action)
                    step_stats.update(
                        {
                            "step_index": t,
                            "action_first7": action[:7].tolist(),
                        }
                    )
                    episode_stats["step_stats"].append(step_stats)
                    executed_actions.append(action)
                    print(
                        "[env.step]",
                        f"step={t}",
                        f"shape={step_stats['shape']}",
                        f"min={step_stats['min']:.6f}",
                        f"max={step_stats['max']:.6f}",
                        f"mean={step_stats['mean']:.6f}",
                        f"std={step_stats['std']:.6f}",
                        f"first7={np.array2string(action[:7], precision=6, separator=', ')}",
                    )

                    obs, reward, done, info = env.step(action.tolist())

                    episode_images.append(img)
                    episode_wrist_images.append(wrist_img)
                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_rewards.append(float(reward))
                    episode_dones.append(bool(done))

                    if done:
                        task_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    episode_stats["exception"] = repr(e)
                    break

            task_episodes += 1

            episode_success = bool(done)
            episode_len = int(len(episode_actions))

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_filename = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4"
            imageio.mimwrite(video_filename, [np.asarray(x) for x in replay_images], fps=10)

            debug_video_filename = DEBUG_VIDEOS_DIR / f"task{task_id}_ep{episode_idx}_{suffix}.mp4"
            if pathlib.Path(video_filename).resolve() != debug_video_filename.resolve():
                shutil.copy2(video_filename, debug_video_filename)

            actions_array = (
                np.stack(executed_actions, axis=0) if executed_actions else np.zeros((0, 7), dtype=np.float32)
            )
            np.save(DEBUG_ROOT / f"actions_task{task_id}_ep{episode_idx}.npy", actions_array)

            episode_stats["success"] = episode_success
            episode_stats["duration"] = int(t)
            episode_stats["num_executed_actions"] = int(actions_array.shape[0])
            episode_stats["video_filename"] = str(video_filename)
            episode_stats["debug_video_filename"] = str(debug_video_filename)

            stats_path = DEBUG_ROOT / f"stats_task{task_id}_ep{episode_idx}.json"
            with stats_path.open("w", encoding="utf-8") as f:
                json.dump(episode_stats, f, indent=2, ensure_ascii=False)

            if args.save_success_only and not episode_success:
                logging.info("Skipping failed episode because --save-success-only is set.")
            else:
                episode_dir = dataset_out_dir / f"task{task_id}_ep{episode_idx}"
                _save_episode_dataset(
                    episode_dir=episode_dir,
                    images=episode_images,
                    wrist_images=episode_wrist_images,
                    states=episode_states,
                    actions=episode_actions,
                    rewards=episode_rewards,
                    dones=episode_dones,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    task_description=task_description,
                    success=episode_success,
                    prompt=prompt,
                )
                saved_episodes += 1
                saved_successes += int(episode_success)
                saved_steps += episode_len

                if args.max_episodes_total is not None and saved_episodes >= args.max_episodes_total:
                    stop_collection = True

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {task_episodes}")
            if task_episodes > 0:
                logging.info(f"# successes: {task_successes} ({task_successes / task_episodes * 100:.1f}%)")

        if task_episodes > 0:
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(
                f"Current total success rate: {float(saved_successes) / float(saved_episodes) if saved_episodes > 0 else 0.0}"
            )

    avg_episode_len = float(saved_steps) / float(saved_episodes) if saved_episodes > 0 else 0.0
    print(
        "[dataset_stats]",
        f"episodes={saved_episodes}",
        f"successes={saved_successes}",
        f"total_steps={saved_steps}",
        f"avg_episode_len={avg_episode_len:.2f}",
    )


def _save_episode_dataset(
    *,
    episode_dir: pathlib.Path,
    images: list[np.ndarray],
    wrist_images: list[np.ndarray],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    dones: list[bool],
    task_id: int,
    episode_idx: int,
    task_description: str,
    success: bool,
    prompt: str,
) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)

    images_array = _stack_or_empty(images, dtype=np.uint8)
    wrist_images_array = _stack_or_empty(wrist_images, dtype=np.uint8)
    states_array = _stack_or_empty(states, dtype=np.float32)
    actions_array = _stack_or_empty(actions, dtype=np.float32)
    rewards_array = np.asarray(rewards, dtype=np.float32)
    dones_array = np.asarray(dones, dtype=np.bool_)

    np.save(episode_dir / "images.npy", images_array)
    np.save(episode_dir / "wrist_images.npy", wrist_images_array)
    np.save(episode_dir / "states.npy", states_array)
    np.save(episode_dir / "actions.npy", actions_array)
    np.save(episode_dir / "rewards.npy", rewards_array)
    np.save(episode_dir / "dones.npy", dones_array)

    meta = {
        "task_id": task_id,
        "episode_idx": episode_idx,
        "task_description": task_description,
        "success": bool(success),
        "episode_len": int(len(actions)),
        "action_dim": int(actions_array.shape[-1]) if actions_array.ndim == 2 and actions_array.shape[0] > 0 else 0,
        "state_dim": int(states_array.shape[-1]) if states_array.ndim == 2 and states_array.shape[0] > 0 else 0,
    }
    with (episode_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with (episode_dir / "prompt.txt").open("w", encoding="utf-8") as f:
        f.write(prompt)


def _stack_or_empty(items: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    if items:
        return np.asarray(np.stack(items, axis=0), dtype=dtype)
    return np.asarray(items, dtype=dtype)


def _print_observation_stats(element: dict) -> None:
    image = np.asarray(element["observation/image"])
    wrist_image = np.asarray(element["observation/wrist_image"])
    state = np.asarray(element["observation/state"])
    prompt = element["prompt"]

    print(
        "[observation/image]",
        f"shape={image.shape}",
        f"dtype={image.dtype}",
        f"min={image.min()}",
        f"max={image.max()}",
    )
    print(
        "[observation/wrist_image]",
        f"shape={wrist_image.shape}",
        f"dtype={wrist_image.dtype}",
        f"min={wrist_image.min()}",
        f"max={wrist_image.max()}",
    )
    print(
        "[observation/state]",
        f"shape={state.shape}",
        f"dtype={state.dtype}",
        f"min={state.min()}",
        f"max={state.max()}",
    )
    print("[prompt]", prompt)


def _array_stats(array: np.ndarray) -> dict:
    array = np.asarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
