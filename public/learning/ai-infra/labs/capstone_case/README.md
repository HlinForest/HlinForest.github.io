# Capstone case: slow and unstable Transformer training

This is a synthetic evidence packet. Treat every artifact as incomplete evidence,
not as an authoritative diagnosis.

## Reported symptoms

- Loss becomes NaN between steps 150 and 300 in mixed precision.
- Eight-GPU throughput is only 3.1x the single-GPU baseline.
- GPU utilization periodically drops near zero.
- Process memory is about 25% above the first spreadsheet estimate.
- One node occasionally hangs in AllReduce.
- A custom extension imports in environment A but not environment B.

## Artifacts

- `model-config.json`: model and training dimensions.
- `train-step.py.txt`: suspicious training-loop excerpt.
- `environment-a.txt`, `environment-b.txt`: two environment snapshots.
- `profile-summary.csv`: simplified per-step profile summary.
- `topology.txt`: simplified eight-GPU topology.
- `nccl-log.txt`: selected NCCL initialization and failure lines.

## Your job

Follow the deliverables and rubric in
`../../lessons/0013-capstone.html`. Separate observed facts, inferences, and
experiments that still need to be run.
