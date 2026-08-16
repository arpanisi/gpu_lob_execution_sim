# Vast.ai Runbook

This project runs from `limitbookorder/gpu_lob_execution_sim`.

## Data And Output Contract

- `data/` holds local or Vast-mounted source data. Its contents are ignored by Git.
- `data/raw/` holds downloaded Bybit trade/order-book archive files.
- `outputs/` holds generated reconstructed events, validation reports, speed measurements, model checkpoints, and evaluation reports. Its contents are ignored by Git.

## Confirm Bybit Source URLs

The locked source is Bybit `BTCUSDT` linear perpetual historical data.

Confirmed trade-print archive pattern:

```text
https://public.bybit.com/trading/BTCUSDT/BTCUSDTYYYY-MM-DD.csv.gz
```

Confirmed order-book archive pattern:

```text
https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/YYYY-MM-DD_BTCUSDT_ob500.data.zip
```

Probe one date and download small byte samples:

```bash
python scripts/probe_bybit_sources.py \
  --date 2024-01-01 \
  --download-samples \
  --sample-bytes 50000 \
  --output outputs/reports/source_probe.json
```

The trade archive timestamps arrive as seconds with decimals. Runtime parsing converts them to integer milliseconds, matching the coding plan's internal timestamp unit.

## Tier 1 Synthetic Validation

This is a local plumbing check for the Step 1/2 reconstruction and replay-validation logic. It writes only generated files under `outputs/`.

```bash
python scripts/run_tier1_synthetic.py \
  --events-output outputs/events/tier1_synthetic_events.jsonl \
  --report-output outputs/reports/tier1_synthetic_validation.json
```

This is not a substitute for the real Bybit Tier 1 reconstruction over 2024-01-01 through 2024-01-10. It exists to verify the event materialization and validation path before the raw order-book ZIP parser is run at scale.

## Real Archive Reconstruction Smoke

After full raw files exist under `data/raw/`, reconstruct a small prefix of real Bybit order-book records and validate replay reproduction:

```bash
python scripts/reconstruct_events.py \
  --orderbook-zip data/raw/2024-01-01_BTCUSDT_ob500.full.data.zip \
  --trades-gzip data/raw/BTCUSDT2024-01-01.csv.gz \
  --max-records 100 \
  --events-output outputs/events/reconstructed_2024-01-01_head.jsonl \
  --report-output outputs/reports/reconstruction_2024-01-01_head.json
```

## Build Reconstructed Event Ranges

Build the real Tier 1 event files from the first 10 locked dates:

```bash
python scripts/build_reconstructed_range.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-10 \
  --raw-dir data/raw \
  --events-dir outputs/events \
  --reports-dir outputs/reports/reconstruction \
  --manifest-output outputs/reports/reconstruction_manifest_tier1.json \
  --download-missing
```

Build the Tier 2 event files from the first 30 locked dates:

```bash
python scripts/build_reconstructed_range.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-30 \
  --raw-dir data/raw \
  --events-dir outputs/events \
  --reports-dir outputs/reports/reconstruction \
  --manifest-output outputs/reports/reconstruction_manifest_tier2.json \
  --download-missing
```

Build the full Tier 3 event files for the locked 2024 calendar year:

```bash
python scripts/build_reconstructed_range.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --raw-dir data/raw \
  --events-dir outputs/events \
  --reports-dir outputs/reports/reconstruction \
  --manifest-output outputs/reports/reconstruction_manifest_tier3.json \
  --download-missing
```

## Tier 1 Replay Smoke

After reconstructed event JSONL exists, run `N=4` one-agent replay over the event stream. This exercises episode windows, observations, agent order insertion, historical replay, matching, rewards, and output reporting.

```bash
python scripts/run_tier1_replay.py \
  --events outputs/events/reconstructed_2024-01-01_head.jsonl \
  --report-output outputs/reports/tier1_replay_report.json \
  --batch-size 4 \
  --max-steps 20
```

The report is written under `outputs/reports/`. A passing smoke has `passed: true` and `observation_lengths: [90]`.

## Step 3 Batched Speed Measurement

Run this on the Vast GPU node after JAX is installed. It measures batched JAX execution against sequential CPU replay for the locked `N=16,64,256` batch sizes and writes the actual ratios.

```bash
python scripts/measure_step3_speedup.py \
  --steps 10000 \
  --batch-sizes 16,64,256 \
  --report-output outputs/reports/step3_speedup.json
```

Local CPU JAX is only for runtime validation of the JAX code path. It is suitable for:

```bash
.venv/bin/python -m pytest tests/test_jax_matching.py tests/test_jax_batched_env.py -q
```

and for a tiny direct batched-environment smoke. Do not use local CPU JAX for Tier 2 or Tier 3 training; the locked Tier 2 shape (`N=64`, rollout length `128`, capacity `1000`) is GPU/Vast scope.

## Tier 2 IPPO Preflight

Before launching Tier 2 training on Vast, confirm the reconstructed event files and locked IPPO configuration are visible. This command does not train.

```bash
python scripts/prepare_tier2_ippo.py \
  --events-dir outputs/events \
  --pattern 'reconstructed_2024-01-*.jsonl' \
  --report-output outputs/reports/tier2_ippo_preflight.json \
  --batch-size 64 \
  --agents 2
```

For a local single-file smoke, use:

```bash
python scripts/prepare_tier2_ippo.py \
  --events-dir outputs/events \
  --pattern 'reconstructed_2024-01-01_head.jsonl' \
  --report-output outputs/reports/tier2_ippo_preflight.json \
  --batch-size 64 \
  --agents 2
```

## Tier 2 IPPO Short Run

Run this on Vast after Step 3 speed measurement has produced `outputs/reports/step3_speedup.json` and the first 30 days of reconstructed event files exist under `outputs/events/`.

```bash
python scripts/run_tier2_ippo.py \
  --events-dir outputs/events \
  --pattern 'reconstructed_2024-01-*.jsonl' \
  --speedup-report outputs/reports/step3_speedup.json \
  --report-output outputs/reports/tier2_ippo_report.json \
  --checkpoint-output outputs/checkpoints/tier2_ippo_params.npz \
  --batch-size 64 \
  --agents 2 \
  --training-steps 4096
```

The report records the three Tier 2 acceptance checks: measured Step 3 speedup ratios for `N=16,64,256`, at least one Step 7 informed-flow proxy firing, and both agents' policy action summaries moving away from initialization during the short run.

## Tier 3 Full-Run Preflight

Before launching the full-year run, verify that Tier 1 passed, Tier 2 passed its acceptance checks, Step 3 speed ratios are present, and the full 2024 reconstructed event set is available.

```bash
python scripts/prepare_tier3_full_run.py \
  --events-dir outputs/events \
  --pattern 'reconstructed_2024-*.jsonl' \
  --tier1-report outputs/reports/tier1_replay_report.json \
  --tier2-report outputs/reports/tier2_ippo_report.json \
  --speedup-report outputs/reports/step3_speedup.json \
  --report-output outputs/reports/tier3_preflight.json \
  --batch-size 256 \
  --agents 2
```

The full Tier 3 launch must only proceed when `ready_for_tier3` is `true`.

## Step 8 Time-Of-Day Report

After evaluation writes the per-episode log, produce the 24-bucket UTC hour-of-day report from that same log.

```bash
python scripts/report_time_of_day.py \
  --evaluation-log outputs/evaluation/tier3_eval_episodes.csv \
  --report-output outputs/reports/time_of_day_breakdown.json
```
