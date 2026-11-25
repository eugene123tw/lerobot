# Training π₀.₅ and ACT on `eugene123tw/pick-blocks-from-slot` (single A100 40 GB)

## 1. Dataset facts (from `meta/info.json`)

| Field          | Value                                                     |
| -------------- | --------------------------------------------------------- |
| Robot          | `SO101_Follower` (6-DoF: 5 joints + gripper)              |
| Episodes       | 96                                                        |
| Total frames   | **52,101**                                                |
| FPS            | 30                                                        |
| Cameras        | `observation.images.overview`, `observation.images.wrist` |
| Image size     | 480×640, AV1 video                                        |
| Quantile stats | `q01` / `q99` present (π₀.₅'s QUANTILES default works)   |

## 2. How epochs map to `--steps`

`lerobot-train` has no `--epochs` flag. One `--steps` iteration consumes **one micro-batch** of `--batch_size`
frames, so:

```text
steps = ceil(epochs × total_frames / batch_size)
```

Gradient accumulation does **not** change this. With `--accelerator.gradient_accumulation.steps=N`, `--steps`
still counts micro-batches: N micro-batches go into each optimizer update, and the LR scheduler
also counts micro-batches.

## 3. Default training precision: fp32, not bf16

By default both settings that control precision are off:

| Setting                           | Default   | Meaning                                       |
| --------------------------------- | --------- | --------------------------------------------- |
| `--policy.dtype` (π₀.₅)           | `float32` | Dtype of the weights, gradients and AdamW state |
| `--accelerator.mixed_precision`   | `no`      | No autocast                                   |
| `--tf32`                          | `true`    | fp32 matmuls/convs use TF32 on Ampere (A100)  |

So by default training runs in **fp32**, with TF32 tensor cores for matmuls on an A100.

## 4. π₀.₅: full fine-tune, 10 epochs

### Why it needs `--policy.dtype=bfloat16` on 40 GB

π₀.₅ has about 3.3B parameters (PaliGemma 2B + SigLIP + 300M action expert). A full fine-tune with AdamW
keeps the weights, gradients and two Adam moments for each parameter:

| Mode                                              | Approx. static memory                    | Fits 40 GB? |
| ------------------------------------------------- | ---------------------------------------- | ----------- |
| fp32 (default): 4+4+8 bytes/param                 | ~53 GB                                   | No          |
| fp32 weights + `--accelerator.mixed_precision=bf16` | ~53 GB (bf16 compute only, fp32 master weights) | No          |
| `--policy.dtype=bfloat16`: 2+2+4 bytes/param      | ~28 GB (vision tower and norms stay fp32) | Yes, ~10 GB left for activations |

`--policy.dtype=bfloat16` is the only way to fully fine-tune on one 40 GB GPU. It is the setting the
LeRobot π₀.₅ recipes use. The AdamW update is applied directly to the bf16 weights; the SigLIP vision
tower, the projector and the layer norms stay in fp32 (see `to_bfloat16_for_selected_params` in
`modeling_pi05.py`). Don't also pass `--accelerator.mixed_precision=bf16`: the weights are already bf16.

`--policy.gradient_checkpointing=true` recomputes the Gemma decoder and SigLIP activations in the
backward pass. This makes each step about 20–30% slower, but uses much less activation memory.

### Batch size: 4 per step, effective batch 32 through accumulation

Expect `--batch_size=32` to OOM. Use micro-batch **4** with **8** accumulation steps, for an effective
batch of 4 × 8 = **32**.

- 10 epochs: `ceil(10 × 52101 / 4)` = **130,253 steps** (16,282 optimizer updates)
- LR warmup: π₀.₅'s default is 1,000 warmup steps, and the scheduler counts micro-batches. With 8
  accumulation steps, the default would give only 125 updates of warmup. Use
  `--policy.scheduler_warmup_steps=8000`, which is 1,000 optimizer updates.
- LR decay: the default `scheduler_decay_steps=30000` only auto-scales when `--steps` is smaller.
  With 130k steps, the LR would reach its floor at 30k and stay there. Set it to `--steps`.

### Command

```bash
hf auth login   # π₀.₅ uses the gated google/paligemma-3b-pt-224 tokenizer: accept its license first
export HF_USER=eugene123tw

lerobot-train \
    --dataset.repo_id=eugene123tw/pick-blocks-from-slot \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.device=cuda \
    --policy.repo_id=${HF_USER}/pi05_pick-blocks-from-slot \
    --batch_size=4 \
    --steps=130253 \
    --save_freq=13000 \
    --num_workers=8 \
    --trackio.enable=true \
    --trackio.project=pick-blocks-from-slot \
    --output_dir=./outputs/pi05_pick-blocks-from-slot \
    --job_name=pi05_pick-blocks-from-slot \ 
```

### Flag details

| Flag                                                        | Why                                                                                   |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `--policy.pretrained_path=lerobot/pi05_base`                | Loads the base weights. The config comes from `--policy.type` plus your flags.          |
| `--policy.dtype=bfloat16`                                   | Halves the memory for weights, gradients and Adam state (see the table above).           |
| `--policy.gradient_checkpointing=true`                      | Trades extra compute for activation memory.                                              |
| `--policy.freeze_vision_encoder=false` / `train_expert_only=false` | Full fine-tune: the VLM, the vision encoder and the action expert are all trained. These are already the defaults; they are set here to make it explicit. |
| `--batch_size=4` + `--accelerator.gradient_accumulation.steps=8` | Effective batch of 32 within 40 GB.                                                |
| `--steps=130253`                                            | 10 epochs at micro-batch 4.                                                             |
| `--policy.scheduler_warmup_steps` / `scheduler_decay_steps` | Cosine schedule (peak 2.5e-5 → 2.5e-6) that spans the whole run.                        |
| `--save_freq=13000`                                         | About one checkpoint per epoch (13,025 steps per epoch). Each checkpoint is large; lower the frequency if disk space is tight. |
| `--num_workers=8`                                           | Decoding AV1 video for two 480×640 cameras uses a lot of CPU. Raise this if `dataloading_s` in the logs is a large share of `step_s`. |
| `--policy.repo_id`                                          | `push_to_hub` defaults to true, so a repo id is needed. Use `--policy.push_to_hub=false` to keep the model local only. |

### If batch 8 fits

Watch `mem_gb` in the training logs. If it stays well under about 34 GB at batch 4, try micro-batch 8.
This gives fewer steps and better GPU use:

```bash
    --batch_size=8 \
    --accelerator.gradient_accumulation.steps=4 \
    --steps=65127 \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=65127 \
    --save_freq=6500 \
```

Optional speedups: `--policy.optimizer_fused=true` (fused AdamW) and `--policy.compile_model=true`
(slow first steps while it compiles, and it uses more memory, so test it first).

## 5. ACT: 30 epochs

### Choosing the batch size

ACT is small (~52M parameters, ResNet-18 backbone). It needs about 2–6 GB at batch 8, so memory is not
the limit on an A100. The real limits are:

1. **Optimization**: ACT's defaults (`lr=1e-5`, constant LR, no scheduler) were tuned for small
   batches (the original ACT used 8). A very large batch with the same LR means fewer and less noisy
   updates. Over a fixed number of epochs, that usually converges worse.
2. **Data loading**: two AV1 480×640 streams per sample. With large batches the GPU waits on video
   decoding.

**Recommendation: `--batch_size=32`.** This is about 4× the reference batch and uses roughly 12–18 GB.
It still gives about 49k updates over 30 epochs, which is in the range where ACT trains well.

- 30 epochs: `ceil(30 × 52101 / 32)` = **48,845 steps**
- Batch 64 also fits, giving 24,423 steps. Only use it if you also raise the LR to about `2e-5`,
  since there are half as many updates.

### Command

```bash
lerobot-train \
    --dataset.repo_id=eugene123tw/pick-blocks-from-slot \
    --policy.type=act \
    --policy.device=cuda \
    --policy.repo_id=${HF_USER}/act_pick-blocks-from-slot \
    --batch_size=32 \
    --steps=48845 \
    --save_freq=10000 \
    --num_workers=12 \
    --trackio.enable=true \
    --trackio.project=pick-blocks-from-slot \
    --output_dir=./outputs/act_pick-blocks-from-slot \
    --job_name=act_pick-blocks-from-slot
```

### Flag details

| Flag                  | Why                                                                                          |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `--policy.type=act`   | Trains from scratch, with an ImageNet-pretrained ResNet-18 as the vision backbone.            |
| `--batch_size=32`     | Balances the number of updates, GPU use and data loading (see above).                         |
| `--steps=48845`       | 30 epochs at batch 32.                                                                        |
| `--num_workers=12`    | ACT's forward pass is cheap, so video decoding is the bottleneck. Use more workers if `dataloading_s` is high. |
| Precision             | Default fp32 (with TF32). Add `--accelerator.mixed_precision=bf16` for a modest speedup if you need it; ACT is usually data-loading bound anyway. |

The ACT defaults that are kept: `chunk_size=100` / `n_action_steps=100` (3.3 s at 30 fps), VAE on
(`kl_weight=10`), and MEAN_STD normalization.

## 6. Monitoring

Every `log_freq` (200) steps the training log prints `loss`, `grad_norm`, `lr`, `mem_gb`,
`dataloading_s`, `update_s` and `step_s`:

- `mem_gb` is the peak GPU memory for the step. Use it to decide whether a bigger π₀.₅ micro-batch fits.
- If `dataloading_s` is close to `step_s`, data loading is the bottleneck: raise `--num_workers`.

### Experiment tracking (Trackio, not W&B)

This version of LeRobot replaced the Weights & Biases logger with
[Trackio](https://huggingface.co/docs/trackio), which has a W&B-style API and dashboard. `--wandb.*` flags
are no longer accepted. Both commands above enable it:

| Flag                                     | Meaning                                                                                       |
| ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| `--trackio.enable=true`                  | Log `loss`, `grad_norm`, `lr`, `mem_gb`, timings and the full train config.                    |
| `--trackio.project=pick-blocks-from-slot` | Groups the π₀.₅ and ACT runs in one project so you can compare them. The default project is `lerobot`. |
| `--trackio.run_name=<name>` (optional)   | Defaults to `<job_name>-<id>`. It is saved in the checkpoint, so `--resume=true` reattaches to the same run. |
| `--trackio.space_id=<user>/<space>` (optional) | Syncs to a Hugging Face Space (created if missing) instead of the local database. You can then view it from anywhere. |
| `--trackio.disable_artifact=false` (optional) | Also upload checkpoint weights as Trackio artifacts. Off by default because π₀.₅ checkpoints are large. |

With the local database (the default), open the dashboard with:

```bash
trackio show --project pick-blocks-from-slot
```

### Monitoring remotely (training machine behind a VPN)

The local database only exists on the training machine. To watch a run without joining the VPN, the
training machine opens an outbound connection: either a public tunnel to its local dashboard (Option 1),
or pushes to a Hugging Face Space (Options 2 and 3). Nothing has to connect to the machine itself.

Common setup on the training machine:

1. Log in with a token that has **write** access: `hf auth login`, or `export HF_TOKEN=hf_...`.
2. Check that the machine can reach the Hub through the VPN or its proxy: `hf auth whoami`.
3. Add these flags to the training command (no `--trackio.space_id`, so metrics go to the local database):

   ```bash
       --trackio.enable=true \
       --trackio.project=pick-blocks-from-slot \
       --trackio.capture_console=false \
       --log_freq=50 \
   ```

#### Option 1 (free, live): public share link to the local dashboard

On the training machine, in a second `tmux` pane next to the training run, start the dashboard with a
share tunnel:

```bash
uv run python -c "import trackio; trackio.show(project='pick-blocks-from-slot', share=True, open_browser=False)"
```

It prints a public URL like `https://<random>.gradio.live/?project=pick-blocks-from-slot`. Open it from
any browser. The dashboard reads the local database directly, so it updates as training logs, every
`--log_freq` steps. No HF account or PRO is needed.

- Keep the process running: the link stops working when it exits. Share links are also temporary; if
  the link expires, run the command again for a new URL.
- Anyone with the URL can see the dashboard. The URL is random, but don't post it publicly.
- If the printed URL contains `write_token=...`, remove that parameter before you share it. The token
  lets anyone who has it write logs to the project.
- The tunnel needs outbound access to the `gradio.live` share servers. If the VPN or firewall blocks
  it, no URL is printed; use Option 2 instead.

#### Option 2 (free): static Space, refreshed periodically

A static Space is free, but it is a **public**, read-only snapshot of the local database. Push the
snapshot from the training machine:

```bash
uv run python -c "import trackio; trackio.sync(project='pick-blocks-from-slot', space_id='eugene123tw/lerobot-trackio-static', sdk='static', force=True)"
```

The dashboard is at https://huggingface.co/spaces/eugene123tw/lerobot-trackio-static. It only changes
when you sync again, so keep it updated with a loop in a second `tmux` pane:

```bash
while true; do
  uv run python -c "import trackio; trackio.sync(project='pick-blocks-from-slot', space_id='eugene123tw/lerobot-trackio-static', sdk='static', force=True)"
  sleep 600
done
```

- `force=True` overwrites the previous snapshot without asking for confirmation.
- Static Spaces can't be private, so everyone can see the metrics and the logged train config.
  `--trackio.capture_console=false` keeps console output and tracebacks out of the snapshot.
- Use a name that isn't already a Gradio Space, such as the failed `eugene123tw/lerobot-trackio`. A
  sync is rejected when the existing Space is a different type.

#### Option 3 (needs HF PRO): live private Gradio Space

Hosting a Gradio Space on free `cpu-basic` hardware requires a PRO subscription. Without one, Space
creation fails with `402 Payment Required` and Trackio falls back to the local database. With PRO, add
these flags for a live dashboard that updates every `--log_freq` steps:

```bash
    --trackio.space_id=${HF_USER}/lerobot-trackio \
    --trackio.private=true \
```

Notes:

- `--log_freq` sets how often metrics are logged (every 200 steps by default). At about 0.9 s per
  π₀.₅ step on an A100 40 GB, 50 steps is a new point roughly every 45 s.
- Run training inside `tmux` or `screen`, or with `nohup`, so that a dropped SSH or VPN session does not
  kill it.

To resume after an interruption:

```bash
lerobot-train \
    --config_path=./outputs/<run>/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```
