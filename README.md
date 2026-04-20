### SP8192 + ED-Fusion + Hessian-Modulated SDClip + AR-Mixed GPTQ + Best-of-N Multi-Codec (brotli / lzma / zstd) + Legal Score-First TTT.

## Approaches

- SP8192, 11L / 512d / 8H / 4KV, MLP 4×, leaky-ReLU(0.5)², partial RoPE, LN scale, tied emb, logit softcap 30
- Depth recurrence L3–5 (on from train frac 0.35); parallel residuals from L7
- QK-gain 5.28; EMA 0.9965; MuonEq-R + AdamW; legal score-first TTT (sliding stride 64)
- GPTQ int6 matrices / int8 emb; byte-shuffle + pick smallest of brotli-11 / lzma-9-extreme / zstd-22 (1-byte tag)
- **Added:** ED-Fusion (scalar gate encoder→decoder before final norm); Hessian-modulated per-row SDClip; AR-mixed GPTQ calib (train shards + self-gen top-k)

## PR credit (inherited stack)


| Piece                  | Credit                                          |
| ---------------------- | ----------------------------------------------- |
| SP8192 + GPTQ + SDClip | @clarkkev — PR #1394                            |
| Depth recurrence       | @dexhunter — PR #1331, #1437                    |
| Parallel residuals     | @Robby955 — PR #1412; @msisovic — PR #1204      |
| Legal TTT              | @abaybektursun — PR #549; @dexhunter — PR #1413 |
| Hyperparams            | @X-Abhishek-X — PR #1445                        |


## Files

`train_gpt.py`, `requirements.txt`, `submission.json`

## Results

## Download SP8192

From the **root of this repository** (the directory that contains `data/`):

```bash
pip install huggingface_hub
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 100
```

Optional: omit `--train-shards` to use the downloader’s default shard count. Override the Hugging Face dataset id only if needed: `MATCHED_FINEWEB_REPO_ID=...` (see default `REPO_ID` in `data/cached_challenge_fineweb.py`).

If you get `**dataset fineweb10B_sp8192 not found in .../manifest.json**`, remove or rename `data/manifest.json` and run the same command again.

Expected layout:

- `data/datasets/fineweb10B_sp8192/`
- `data/tokenizers/` (SentencePiece model for SP8192)
- `data/manifest.json`

Training defaults: `DATA_PATH=./data/datasets/fineweb10B_sp8192`, `TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model`, `VOCAB_SIZE=8192`.

## Run

From the **root of this repository**:

```bash
pip install brotli zstandard sentencepiece

SEED=42 \
  QK_GAIN_INIT=5.28 \
  ED_FUSION_ENABLED=1 ED_FUSION_INIT=-4.0 \
  MATRIX_HESSIAN_ALPHA=0.25 \
  GPTQ_AR_SELF_CALIB_FRAC=0.5 GPTQ_AR_TOPK=64 \
  COMPRESSOR=auto \
  TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-20_MultiTrick_EDFusion_HessianSDClip_BestOfN/train_gpt.py
```

**2× RTX 4090 (smaller batch):** `WORLD_SIZE` must divide 8. Lower `TRAIN_BATCH_TOKENS` / `VAL_BATCH_SIZE` so each rank fits in VRAM; keep `TRAIN_BATCH_TOKENS` divisible by `WORLD_SIZE × (8 // WORLD_SIZE) × TRAIN_SEQ_LEN`.

```bash
SEED=42 \
  QK_GAIN_INIT=5.28 \
  ED_FUSION_ENABLED=1 ED_FUSION_INIT=-4.0 \
  MATRIX_HESSIAN_ALPHA=0.25 \
  GPTQ_AR_SELF_CALIB_FRAC=0.5 GPTQ_AR_TOPK=64 \
  COMPRESSOR=auto \
  TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  MAX_WALLCLOCK_SECONDS=2400 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

