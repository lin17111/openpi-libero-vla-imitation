# Result Table

| Stage | Setup | Offline Metric | Closed-loop Metric | Notes |
|---|---|---:|---:|---|
| Teacher π0.5 | `libero_spatial` | `492/500` | `98.4%` | Benchmark teacher rollout |
| Single-step student | `image + wrist_image + state -> 7D action` | `val_loss ≈ 0.0143` | `0/10` | Compounding error in closed loop |
| Action-chunk student | `image + wrist_image + state -> 10x7 action chunk` | `val_loss ≈ 0.0152` | `35/50` (`70.0%`) | Much better rollout robustness |
