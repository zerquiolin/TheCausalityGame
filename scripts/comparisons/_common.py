"""Shared artifact and observation helpers for comparison scripts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from TheCausalityGame.core.contracts.dto.agent import RoundObservation
from TheCausalityGame.core.contracts.dto.environment import (
    RoundInfo,
    Samples,
    SamplesCollection,
)
from TheCausalityGame.core.infrastructure.decisions import Decision


def input_digest(data: pd.DataFrame, *arrays: object) -> str:
    """Return a stable digest proving paired implementations received identical inputs."""
    digest = hashlib.sha256()
    digest.update(json.dumps(list(data.columns), default=str).encode())
    digest.update(json.dumps([str(dtype) for dtype in data.dtypes]).encode())
    digest.update(pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes())
    for value in arrays:
        digest.update(json.dumps(value, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate numeric metrics by method and implementation for the CSV report."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("method", "")), str(row.get("implementation", "")))
        groups.setdefault(key, []).append(row)

    aggregate: list[dict[str, Any]] = []
    for (method, implementation), members in sorted(groups.items()):
        record: dict[str, Any] = {
            "method": method,
            "implementation": implementation,
            "runs": len(members),
            "successes": sum(member.get("status") == "ok" for member in members),
            "failures": sum(member.get("status") != "ok" for member in members),
        }
        numeric_keys = sorted(
            {
                key
                for member in members
                for key, value in member.items()
                if key != "seed"
                and isinstance(value, int | float)
                and not isinstance(value, bool)
            }
        )
        for key in numeric_keys:
            values = [
                float(member[key])
                for member in members
                if isinstance(member.get(key), int | float)
                and not isinstance(member.get(key), bool)
            ]
            if values:
                record[f"mean_{key}"] = sum(values) / len(values)
        backends = [
            member.get("backend") for member in members if member.get("backend")
        ]
        if backends:
            record["backend"] = json.dumps(backends[0], sort_keys=True)
        aggregate.append(record)
    return aggregate


def observation(
    data: pd.DataFrame,
    *,
    round_number: int = 1,
    kind: str = "observational",
    interventions: dict[str, Any] | None = None,
) -> RoundObservation:
    """Build one inferer update shared by both sides of a comparison."""
    sample = Samples(kind=kind, n=len(data), data=data, interventions=interventions)
    return RoundObservation(
        round_info=RoundInfo(round=round_number),
        decision=Decision.experiment().add_experiment(
            treatment=interventions, n=len(data)
        ),
        samples=SamplesCollection([sample]),
    )


def buffered_rounds(
    data: pd.DataFrame, rounds: int
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Return deterministic round chunks and their cumulative buffers."""
    if rounds < 1 or rounds > len(data):
        message = f"rounds must be between 1 and the sample count ({len(data)})."
        raise ValueError(message)
    indices = np.array_split(np.arange(len(data)), rounds)
    result: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    observed: list[pd.DataFrame] = []
    for round_number, positions in enumerate(indices, start=1):
        chunk = data.iloc[positions].copy()
        observed.append(chunk)
        result.append(
            (round_number, chunk, pd.concat(observed, axis=0, ignore_index=True))
        )
    return result


def output_directory(root: Path | None = None) -> Path:
    """Create a timestamped comparison artifact directory."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = (root or Path("runs/comparisons")) / timestamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_reports(rows: list[dict[str, Any]], directory: Path, stem: str) -> None:
    """Write JSONL, CSV, and a compact Markdown table."""
    if not rows:
        return
    jsonl = directory / f"{stem}.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )
    aggregate = _aggregate(rows)
    columns = sorted({key for row in aggregate for key in row})
    with (directory / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(aggregate)
    columns = sorted({key for row in rows for key in row})
    visible = [
        name
        for name in (
            "case",
            "method",
            "implementation",
            "round",
            "samples_seen",
            "status",
            "shd",
            "pehe",
            "runtime_s",
        )
        if name in columns
    ]
    lines = [
        f"# {stem.replace('_', ' ').title()}",
        "",
        "| " + " | ".join(visible) + " |",
        "| " + " | ".join("---" for _ in visible) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(name, "")) for name in visible) + " |"
        for row in rows
    )
    (directory / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
