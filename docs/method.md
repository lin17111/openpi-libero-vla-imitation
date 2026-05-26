# Method

This project implements a simple teacher-student imitation learning pipeline on top of OpenPI and LIBERO.

## Teacher Policy

- Teacher: `π0.5` checkpoint (`pi05_libero`)
- Environment: `LIBERO spatial`
- Role: generate high-quality teacher rollouts in MuJoCo simulation

## Trajectory Collection

The teacher policy is rolled out in the LIBERO environment and successful trajectories are saved step-by-step.

Saved fields:

- `images.npy`
- `wrist_images.npy`
- `states.npy`
- `actions.npy`
- `rewards.npy`
- `dones.npy`
- `meta.json`
- `prompt.txt`

## Single-step Behavior Cloning

The first student baseline learns a direct mapping:

`image + wrist_image + state → 7D action`

This is a standard offline regression baseline. It is useful as a sanity check, but it can suffer from compounding error in closed loop.

## Action-chunk Behavior Cloning

The second student baseline predicts a short action sequence:

`image + wrist_image + state → 10 x 7 action chunk`

The idea is to encourage temporal consistency over a short horizon, rather than matching only the next action.

## Closed-loop Evaluation

Both student policies are evaluated directly in LIBERO / MuJoCo rollouts.

The evaluation measures whether offline imitation actually transfers to execution success.

## Why Action Chunks Are More Stable

Single-step BC can drift because each prediction is conditioned on a state that depends on previous predictions.

Action-chunk BC helps because:

- the model predicts a temporally coherent segment rather than a one-step impulse;
- the action sequence is less sensitive to small one-step errors;
- rollout behavior is closer to a short-horizon plan than a purely reactive controller.

