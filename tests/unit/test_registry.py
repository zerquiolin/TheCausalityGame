from __future__ import annotations

import pytest

from TheCausalityGame.core.contracts.errors import LoadError
from TheCausalityGame.core.infra.registry import (
    build_from_spec,
    load_class,
    load_entry_point,
)


def test_load_class_disallowed_package_rejected() -> None:
    with pytest.raises(LoadError):
        load_class("notallowed.module:ClassName")


def test_load_entry_point_missing_raises() -> None:
    with pytest.raises(LoadError):
        load_entry_point("tcg.nonexistent", "nope")


def test_build_from_spec_requires_class() -> None:
    with pytest.raises(LoadError):
        build_from_spec({})
