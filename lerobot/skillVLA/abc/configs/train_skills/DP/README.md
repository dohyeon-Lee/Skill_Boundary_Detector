# ABC DP history encoders

Select `dp_history_encoder` in `dp_config.yaml`:

| Value | Training | Inference and SBD |
| --- | --- | --- |
| `flat` | Flatten a fixed history window | Encode the latest window |
| `transformer` | Fixed history window, last token output | Encode the latest window |
| `transformer_cls` | Fixed history window, CLS token output | Encode the latest window |
| `GRU_iid` | Random windows, zero initial hidden for each window | Consume each new state once and carry hidden |
| `GRU_epi` | Chronological chunks within each episode, carry hidden and detach between chunks | Consume each new state once and carry hidden |

Both GRU variants have **2 layers and 128 hidden units**. Plain `gru` is an
alias for `GRU_iid`; spelling is case-insensitive. For IID training,
`dp_n_obs_steps` is the window length. For episode training, it is the
truncated-backpropagation chunk length; hidden retains information from
earlier chunks. Each batch item belongs to a distinct episode stream.
Episode starts and chunk-boundary offsets are randomized as streams refill. A short
first chunk consumes the actual episode prefix without repeated padding,
then full chunks continue without overlap. Thus a boundary phase can include
frames `2..61` followed by `62..121`, rather than always using multiples of 60.
The sampler respects task/episode selection and incomplete action-tail removal.
Streams continue across loader passes, replenishing finished episodes immediately
instead of draining to tiny final batches. Every batch has the configured size
and distinct episode IDs; this requires at least that many usable episodes.
Episodes with fewer starts are preferred when replenishing streams. Long episodes
finish their remaining chunks, and loader-pass boundaries do not reset their hidden.
`GRU_epi` currently supports the offline ABC native dataset in one training
process (one GPU), as used by `submit_train_dp.sh`.
With data-loading workers enabled, the ABC launcher automatically keeps them
alive across `GRU_epi` passes to avoid repeatedly spawning workers on its short
passes. This also applies when resuming a checkpoint through the launcher;
an already running training process keeps its original loader settings.

Training windows/chunks still produce one diffusion action-target sample each.
IID and episode training therefore differ in temporal correlations and sampling
order even with the same batch size. IID inference carries history beyond its
training window; the effect on prediction quality needs experimental evaluation.

At inference, `select_action` updates hidden on **every environment step**,
including while executing an already predicted action chunk. Call `reset()`
at the start of each episode. SBD reconstructs hidden from frame zero and
consumes intermediate frames even when `replan_interval` skips them. No hidden
is shared between episodes, probe samples, or experiments.

Run folders include `_gru_iid` or `_gru_epi` so variants cannot resume each
other's checkpoints. Existing transformer/small run names are preserved.
`dp_unet_size` independently selects `small` (`[128,256,512]`) or `base`
(`[512,1024,2048]`); base omits the `_small` suffix.
Checkpoints record both GRU training and inference modes. Existing checkpoints
without the stateful flag retain their windowed inference behavior.
