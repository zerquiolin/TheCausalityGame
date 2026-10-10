"""Check both documented CLI forms without running expensive manifests."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from TheCausalityGame.core.cli import main as cli_module


@pytest.mark.parametrize("form", ["direct", "subcommand", "global_option"])
def test_cli_routes_manifest_and_output_directory(
    tmp_path: Path, monkeypatch, form: str
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    output = tmp_path / "artifacts"
    execute = Mock()
    monkeypatch.setattr(cli_module, "_execute_run", execute)
    if form == "subcommand":
        args = ["run", str(manifest), "--run-dir", str(output)]
    elif form == "global_option":
        args = ["--run-dir", str(output), str(manifest)]
    else:
        args = [str(manifest), "--run-dir", str(output)]
    result = CliRunner().invoke(cli_module.app, args)
    assert result.exit_code == 0, result.output
    execute.assert_called_once_with(manifest, output)


def test_cli_invalid_json_returns_failure_before_execution(tmp_path: Path) -> None:
    manifest = tmp_path / "broken.json"
    manifest.write_text("not json")
    result = CliRunner().invoke(cli_module.app, ["run", str(manifest)])
    assert result.exit_code == 1
    assert "Unable to parse JSON" in result.output
