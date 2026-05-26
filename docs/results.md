# Results

This project reports both offline imitation metrics and closed-loop rollout metrics.

## Teacher Baseline

The teacher policy `π0.5` on `libero_spatial` achieves:

- total episodes: `500`
- successes: `492`
- success rate: `98.4%`

This provides the source behavior for the student imitation experiments.

## Student Offline Training

### Single-step student

- input: `image + wrist_image + state`
- output: `7D action`
- offline validation loss: about `0.0143`

### Action-chunk student

- input: `image + wrist_image + state`
- output: `10x7 action chunk`
- offline validation loss: about `0.0152`

The offline losses are close, which shows that a similar regression score does not automatically imply similar rollout quality.

## Student Closed-loop Evaluation

### Single-step student

- closed-loop eval: `0/10` success

This is the clearest example of compounding error.

### Action-chunk student

- closed-loop eval: `35/50` success
- success rate: `70.0%`
- average episode length: `139.68`

This is a much stronger rollout result than the single-step baseline.

## Interpretation

- Teacher policy provides strong demonstrations.
- Single-step BC can fit the teacher actions offline but still fail in control.
- Action-chunk BC gives the student a more temporally consistent prediction target and improves closed-loop performance.
- Offline MSE is necessary for debugging, but not sufficient for rollout success.

Rollout visualizations will be added later.

## Conclusion

The current pipeline shows that a lightweight action-chunk student is a better imitation target than a single-step action regressor for this LIBERO / MuJoCo setting.

