# Experiment Summary

## 1. Teacher π0.5 rollout evaluation

- task suite: `libero_spatial`
- total episodes: `500`
- successes: `492`
- success rate: `98.4%`

## 2. Teacher trajectory collection

- successful episodes: `488`
- total steps: `50713`
- average episode length: `103.92`
- saved fields: `images.npy`, `wrist_images.npy`, `states.npy`, `actions.npy`, `rewards.npy`, `dones.npy`, `meta.json`, `prompt.txt`

## 3. Single-step student BC

- input: `image + wrist_image + state`
- output: `7D action`
- offline val_loss: about `0.0143`
- closed-loop eval: `0/10` success
- conclusion: single-step BC suffers from compounding error

## 4. Action-chunk student BC

- input: `image + wrist_image + state`
- output: `10x7 action chunk`
- offline val_loss: about `0.0152`
- closed-loop eval: `35/50` success
- success rate: `70.0%`
- avg episode length: `139.68`

## 5. Main conclusion

- Teacher π0.5 achieves high benchmark success.
- Single-step student can fit actions offline but fails in closed loop.
- Action-chunk imitation significantly improves closed-loop control.
- The teacher rollout → trajectory collection → student training → closed-loop student evaluation pipeline is complete.
