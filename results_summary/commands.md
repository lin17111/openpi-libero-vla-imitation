# Commands

## Teacher rollout

```bash
uv run /home/lin17/openpi/scripts/serve_policy.py --env LIBERO
python /home/lin17/openpi/examples/libero/main.py
```

## Trajectory collection

```bash
python scripts/collect_trajectories.py
```

## Dataset inspection

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/inspect_collected_dataset.py --dataset-dir results/libero_dataset_500 --seed 7 --sample-episodes 3
```

## Single-step student training

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train_student_bc.py --dataset-dir results/libero_dataset_500 --save-dir results/libero_student_bc --epochs 20 --batch-size 64 --lr 1e-3 --num-workers 4 --val-ratio 0.1
```

## Single-step student evaluation

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/eval_student_bc.py --model-path results/libero_student_bc/best_model.pt --task-suite-name libero_spatial --seed 7
```

## Action-chunk student training

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train_student_chunk_bc.py --dataset-dir results/libero_dataset_500 --save-dir results/libero_student_chunk_bc --chunk-size 10 --epochs 20 --batch-size 64 --lr 1e-3 --num-workers 4 --val-ratio 0.1
```

## Action-chunk student evaluation

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/eval_student_chunk_bc.py --model-path results/libero_student_chunk_bc/best_model.pt --task-suite-name libero_spatial --seed 7 --chunk-size 10
```
