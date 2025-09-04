"""Minimal CLI for validating manifests and inspecting structure.

The manifest now uses:
- run_plan.execution: 'sequential' or 'parallel'
- run_plan.parallel_backend: 'thread' or 'process'
- run_plan.max_workers: optional int (None => auto)
- metric_specs split into 'behavior' and 'result', plus 'custom_metric_specs' list
"""

from __future__ import annotations

import json
import pathlib

import typer

from TheCausalityGame.core.contracts.dto import ProblemInstance
from TheCausalityGame.core.infra.error_handling import (
    format_user_message,
    log_exception,
)
from TheCausalityGame.core.infra.logging_ import configure_logging, get_logger
from TheCausalityGame.core.infra.serialization import loads
from TheCausalityGame.core.infra.settings import RuntimeSettings

app = typer.Typer(add_completion=False, help="The Causality Game CLI (Foundations)")


@app.command(help="Validate a manifest (JSON-only).")
def validate(manifest_path: str) -> None:
    """Validate a ProblemInstance JSON file."""
    try:
        data = pathlib.Path(manifest_path).read_text(encoding="utf-8")
        ProblemInstance.model_validate(loads(data))
        typer.secho("OK: manifest is valid.", fg=typer.colors.GREEN)
    except Exception as e:  # noqa: BLE001
        typer.secho(f"Invalid manifest: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command(help="Show parsed manifest.")
def show(manifest_path: str) -> None:
    """Parse and pretty-print a manifest JSON file."""
    data = pathlib.Path(manifest_path).read_text(encoding="utf-8")
    m = ProblemInstance.model_validate(loads(data))
    typer.echo(json.dumps(m.model_dump(), indent=2, ensure_ascii=False))


@app.command(help="Print quick tips about execution and metrics fields.")
def tips() -> None:
    """Print a short guide for the new run_plan/metrics fields."""
    msg = (
        "Tips:\n"
        "- run_plan.execution: 'sequential' runs one agent after another; "
        "'parallel' runs agents concurrently in isolated environments.\n"
        "- run_plan.parallel_backend: 'thread' (I/O/pythonic) or 'process' (CPU-bound sampling).\n"
        "- run_plan.max_workers: cap concurrency (None => auto).\n"
        "- metric_specs: provide 'behavior' (process metric) and 'result' (quality metric).\n"
        "- custom_metric_specs: extra metrics beyond the two canonical ones."
    )
    typer.echo(msg)


@app.command(help="Print short greeting and hints.")
def hello() -> None:
    """Print CLI help hint."""
    typer.echo(
        "TheCausalityGame foundations ready. "
        "Try `tcg validate examples/infra/manifest_template.json` and `tcg tips`."
    )


@app.command(help="Run a manifest.")
def run(
    manifest_path: str,
    runs_dir: str = typer.Option("runs", help="Output directory."),
    mode: str = typer.Option("restricted", help="'restricted' or 'dev'"),
    debug: bool = typer.Option(False, help="Enable debug logs + stacktraces."),
    trusted: bool = typer.Option(None, help="Override callable deliverables policy."),
):
    settings = RuntimeSettings.from_sources(mode=mode, debug=debug, trusted=trusted)
    configure_logging(debug=settings.debug)
    logger = get_logger("tcg.cli")

    try:
        data = pathlib.Path(manifest_path).read_text(encoding="utf-8")
        manifest = ProblemInstance.model_validate(loads(data))
        # hand off to orchestrator (Phase 2)
        # orchestrator.run_all_agents(manifest, runs_dir=runs_dir, settings=settings)
        typer.secho(
            f"Ready to run (mode={settings.mode}, debug={settings.debug}, trusted={settings.trusted})",
            fg=typer.colors.GREEN,
        )
    except Exception as exc:  # noqa: BLE001
        log_exception(
            logger,
            exc,
            context={"cmd": "run", "path": manifest_path},
            debug=settings.debug,
        )
        msg = format_user_message(
            exc, debug=settings.debug, context={"path": manifest_path}
        )
        typer.secho(f"[{msg.code}] {msg.title}: {msg.message}", fg=typer.colors.RED)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
