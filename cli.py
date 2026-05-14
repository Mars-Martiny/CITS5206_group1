"""Click CLI for unified training, batch inference, and leaderboard reporting."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import click
import pandas as pd
import torch

from api import get_leaderboard, get_model_details, infer, train


def _arch_display(model_type_raw: object) -> str:
    mt = str(model_type_raw).lower().strip().replace(" ", "_")
    if mt == "bert":
        return "bert"
    if mt == "tf_idf":
        return "tf_idf"
    if mt == "looped_transformer":
        return "looped_transformer"
    if mt == "bigru" or mt.startswith("bigru_"):
        return "bigru"
    return str(model_type_raw)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Incident text classification toolchain."""


@cli.command("train")
@click.option("--train", "train_csv", required=True, type=click.Path(exists=True))
@click.option("--valid", "valid_csv", required=True, type=click.Path(exists=True))
@click.option("--test", "test_csv", required=True, type=click.Path(exists=True))
@click.option("--model-type", "model_kind", required=True, type=click.Choice(["energy", "damage"]))
@click.option(
    "--architecture",
    required=True,
    type=click.Choice(["tf_idf", "bigru", "bert", "looped_transformer"]),
)
@click.option("--text-col", default="description", show_default=True)
@click.option("--epochs", type=int, default=None)
@click.option("--lr", type=float, default=None)
@click.option("--patience", type=int, default=None)
@click.option("--hidden-dim", type=int, default=None)
@click.option("--batch-size", type=int, default=None)
@click.option(
    "--embedding-type",
    type=click.Choice(["none", "static", "contextual"]),
    default=None,
    help="BiGRU-only embedding mode.",
)
@click.option("--num-loops", type=int, default=None)
@click.option("--fine-tune", is_flag=True, default=False)
def train_models(
    train_csv: str,
    valid_csv: str,
    test_csv: str,
    model_kind: str,
    architecture: str,
    text_col: str,
    epochs: int | None,
    lr: float | None,
    patience: int | None,
    hidden_dim: int | None,
    batch_size: int | None,
    embedding_type: str | None,
    num_loops: int | None,
    fine_tune: bool,
) -> None:
    """Fine-tune a single architecture on the supplied labelled splits."""
    cfg: dict = {}
    if epochs is not None:
        cfg["epochs"] = epochs
    if patience is not None:
        cfg["patience"] = patience
    if hidden_dim is not None:
        cfg["hidden_dim"] = hidden_dim
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    if embedding_type is not None:
        cfg["embedding_type"] = embedding_type
    if num_loops is not None:
        cfg["num_loops"] = num_loops
    if fine_tune:
        cfg["fine_tune"] = True

    if lr is not None:
        if architecture == "bert":
            cfg["learning_rate"] = lr

        elif architecture != "bert":

            def optimizer_fn(model):  # type: ignore[no-untyped-def]
                return torch.optim.Adam(model.parameters(), lr=lr)

            cfg["optimizer_fn"] = optimizer_fn

    summary = train(
        train_path=train_csv,
        valid_path=valid_csv,
        test_path=test_csv,
        model_type=model_kind,
        architecture=architecture,
        train_config=cfg or None,
        text_col=text_col,
    )

    save_dir = Path(summary["config"]["save_dir"]).resolve()
    best_value = summary.get("best_metric_value")

    click.echo(f"Artifacts directory: {save_dir}")
    if best_value is not None:
        click.echo(f"Best tracked metric: {best_value}")


@cli.command("infer")
@click.option("--dataset", required=True, type=click.Path(exists=True))
@click.option("--energy-model", "energy_model", type=str, default=None)
@click.option("--damage-model", "damage_model", type=str, default=None)
@click.option("--output", required=True, type=click.Path())
@click.option("--text-col", default="description", show_default=True)
def infer_dataset(
    dataset: str,
    energy_model: str | None,
    damage_model: str | None,
    output: str,
    text_col: str,
) -> None:
    """Run batch inference and write a CSV with confidence tiers and actions."""
    if energy_model is None and damage_model is None:
        raise click.UsageError("Provide at least one of --energy-model or --damage-model.")

    frame = infer(
        dataset_path=dataset,
        energy_model_dir=energy_model,
        damage_model_dir=damage_model,
        output_path=output,
        text_col=text_col,
    )

    click.echo(f"Rows scored: {len(frame)}")

    if "energy_score" in frame.columns:
        counts = Counter(frame["energy_score"].dropna().astype(str).tolist())
        click.echo(f"Energy tier counts: {dict(counts)}")

    if "damage_score" in frame.columns:
        counts = Counter(frame["damage_score"].dropna().astype(str).tolist())
        click.echo(f"Damage tier counts: {dict(counts)}")

    if "fatal_flag" in frame.columns:
        fatals = (frame["fatal_flag"] == "YES").sum()
        click.echo(f"Fatal-flagged rows: {int(fatals)}")


@cli.command("metrics")
@click.option("--sort-by", default="val_f1_macro", show_default=True)
@click.option(
    "--model-type",
    "model_kind",
    type=click.Choice(["energy", "damage"]),
    default=None,
)
@click.option("--architecture", "architecture", type=str, default=None)
@click.option("--top", default=20, show_default=True, type=int)
@click.option("--model-dir", "model_dir", type=click.Path(exists=True), default=None)
@click.option("--ascending", is_flag=True, default=False)
def metrics_view(
    sort_by: str,
    model_kind: str | None,
    architecture: str | None,
    top: int,
    model_dir: str | None,
    ascending: bool,
) -> None:
    """Inspect leaderboard rows or dump a single run summary."""
    if model_dir is not None:
        details = get_model_details(model_dir)
        click.echo(json_dumps_pretty(details))
        return

    board = get_leaderboard(
        sort_by=sort_by,
        ascending=ascending,
        model_type_filter=model_kind,
        architecture_filter=architecture,
    )

    columns = [
        "timestamp",
        "model_type",
        "architecture",
        "energy_model",
        "val_f1_macro",
        "test_f1_macro",
        "val_accuracy",
        "training_time_sec",
        "model_path",
    ]

    display = board.head(top).copy()
    display["architecture"] = display["model_type"].map(_arch_display)

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(show_header=True, header_style="bold")
        for name in columns:
            table.add_column(name)

        for _, row in display.iterrows():
            table.add_row(*[_fmt_cell(row.get(col)) for col in columns])

        Console().print(table)

    except ImportError:
        trimmed = display[columns].copy()

        with pd.option_context("display.max_columns", None, "display.width", 220):
            click.echo(trimmed.to_string(index=False))


def _fmt_cell(value: object) -> str:
    """Format NaN and None as empty strings for table cells."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def json_dumps_pretty(details: dict) -> str:
    """Serialise dictionaries for readable CLI dumps."""
    return json.dumps(details, indent=2, default=str)


if __name__ == "__main__":
    cli()
