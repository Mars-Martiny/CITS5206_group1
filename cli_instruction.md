# CLI Usage

```
python cli.py [COMMAND] [OPTIONS]
```

Global help: `python cli.py -h`

---

## `train` — Fine-tune a model

```
python cli.py train \
  --train data/train.csv \
  --valid data/valid.csv \
  --test  data/test.csv \
  --model-type  <energy|damage> \
  --architecture <tf_idf|bigru|bert|looped_transformer>
```

**Optional flags:**

| Flag | Description |
|---|---|
| `--text-col TEXT` | Column name for input text (default: `description`) |
| `--epochs N` | Number of training epochs |
| `--lr FLOAT` | Learning rate |
| `--patience N` | Early-stopping patience |
| `--hidden-dim N` | Hidden layer size |
| `--batch-size N` | Mini-batch size |
| `--embedding-type` | BiGRU only: `none`, `static`, or `contextual` |
| `--num-loops N` | Looped transformer only: number of loops |
| `--fine-tune` | Enable fine-tuning mode (flag, no value needed) |

Prints the artifacts directory and best tracked metric on completion.

---

## `infer` — Batch inference

```
python cli.py infer \
  --dataset data/unlabelled.csv \
  --output  results.csv \
  [--energy-model path/to/energy_run] \
  [--damage-model path/to/damage_run]
```

At least one of `--energy-model` or `--damage-model` must be supplied. The output CSV includes confidence tiers and action columns. Prints row counts and tier distributions.

---

## `metrics` — Inspect the leaderboard

```
# Show top 20 runs sorted by val_f1_macro
python cli.py metrics

# Filter and sort
python cli.py metrics --model-type energy --architecture bert --sort-by test_f1_macro --top 10

# Dump a single run's full details
python cli.py metrics --model-dir path/to/run
```

**Optional flags:**

| Flag | Description |
|---|---|
| `--sort-by COL` | Column to sort by (default: `val_f1_macro`) |
| `--ascending` | Sort ascending instead of descending |
| `--model-type` | Filter by `energy` or `damage` |
| `--architecture` | Filter by architecture name |
| `--top N` | Number of rows to show (default: `20`) |
| `--model-dir PATH` | Dump JSON details for a single saved run |

Renders a rich table if the `rich` package is installed, otherwise falls back to plain text.
