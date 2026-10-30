# ArchetypeTrader (Stage 1/2/3 scaffolding)

This folder contains new components for ArchetypeTrader.

## Stage 1: DP demonstrations (paper defaults)

Paper-aligned defaults:
- horizon `h = 72`
- number of DP trajectories `n = 30,000`
- actions fixed to 3 classes: `0=short, 1=flat, 2=long`

### 1.1 Build DP demonstration dataset

Files:
- `dp_planner.py`: Algorithm 1 single-trade DP planner.
- `build_demo_dataset.py`: sample horizons from `data/<dataset>/whole/train.feather` and build DP demo dataset.

Default output:
- `./data/ETHUSDT/archetype/demo_dp_h72_n30000.npz`

### 1.2 Train VQ encoder-decoder (archetype discovery)

Files:
- `../model/vq_archetype.py`: LSTM encoder + VQ codebook + LSTM decoder.
- `train_vq.py`: train VQ model on DP trajectories.

Default outputs:
- checkpoints: `./result/archetype/ETHUSDT/vq/vq_archetype_epoch*.pt` and `vq_archetype_final.pt`
- code assignments: `./data/ETHUSDT/archetype/code_idx_h72_n30000.npy`

The macro-level selector (Stage 2) will pick the archetype code **once per horizon**.

## Stage 2 (minimal runnable): Macro-level selector

This repo now includes a minimal DQN selector that chooses one archetype code per horizon.

Files:
- `env_stage2.py`: `MacroSelectorEnv` (one-step episode; reward is horizon PnL from decoded actions)
- `train_stage2_selector.py`: minimal DQN trainer

Output:
- `./result/archetype/ETHUSDT/stage2/selector.pt`

## Stage 3

- Micro-level adapter policy (TODO)
