"""Typer-based CLI entrypoint for The Causality Game."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import ValidationError

from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.runtime.runner import Runner

app = typer.Typer(
    add_completion=False,
    help="Command-line tools for running The Causality Game problem instances.",
)


def _load_problem_instance(problem_path: Path) -> ProblemInstanceSpec:
    """
    Load and validate a problem instance from disk.

    Parameters
    ----------
    problem_path:
        Path to the JSON problem specification.
    """
    try:
        raw_spec = json.loads(problem_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        typer.secho(f"Problem instance file not found: {problem_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except json.JSONDecodeError as exc:
        typer.secho(
            f"Unable to parse JSON problem instance at {problem_path}:\n{exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        typer.secho(
            f"Unable to read problem instance at {problem_path}: {exc.strerror or exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc

    try:
        return ProblemInstanceSpec.model_validate(raw_spec)
    except ValidationError as exc:
        typer.secho("Problem instance specification is invalid:", fg=typer.colors.RED, err=True)
        typer.echo(exc, err=True)
        raise typer.Exit(code=1) from exc


def _execute_run(problem_path: Path, run_dir: Path) -> None:
    """Shared execution logic used by both the run command and direct invocation."""
    problem_instance = _load_problem_instance(problem_path)

    run_dir.mkdir(parents=True, exist_ok=True)

    typer.secho(
        f"Running problem instance '{problem_instance.id}' (schema {problem_instance.schema_version})",
        fg=typer.colors.GREEN,
    )

    runner = Runner(run_dir=run_dir, problem_instance=problem_instance)

    try:
        runner.run()
    except Exception as exc:  # noqa: BLE001
        typer.secho(
            f"Execution failed with an unexpected error: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc


@app.command()
def run(
    problem_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the problem instance JSON file.",
    ),
    run_dir: Path = typer.Option(
        Path("runs"),
        "--run-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory where run artifacts will be stored.",
    ),
) -> None:
    """
    Execute a problem instance definition.

    The command loads a problem instance specification and runs it using the core runner,
    matching the behaviour of the legacy scripts/main.py helper.
    """
    _execute_run(problem_path, run_dir)


@app.callback(invoke_without_command=True, help="Run a problem instance from a JSON specification.")
def main(
    ctx: typer.Context,
    problem_path: Path | None = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the problem instance JSON file.",
    ),
    run_dir: Path = typer.Option(
        Path("runs"),
        "--run-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory where run artifacts will be stored.",
    ),
) -> None:
    """
    Support invoking `tcg` either with the explicit `run` subcommand or directly as a shortcut.
    """
    if ctx.invoked_subcommand is not None:
        return

    if problem_path is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    _execute_run(problem_path, run_dir)


if __name__ == "__main__":
    app()
