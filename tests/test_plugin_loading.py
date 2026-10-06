"""Load the plugin the way hermes does: as a package under a synthetic name,
including from a profile directory that symlinks the plugin checkout."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMODULES = ["health", "backup", "cli", "importer"]


def _load_like_hermes(plugin_dir: Path, name: str):
    """Mirror hermes' plugins/memory loader: package under a synthetic name,
    then each submodule imported as <name>.<sub>."""
    spec = importlib.util.spec_from_file_location(
        name, plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def isolated(monkeypatch):
    # Top-level fallbacks must not mask a broken relative import.
    for top in SUBMODULES:
        monkeypatch.setitem(sys.modules, top, None)  # top-level import -> ImportError
    monkeypatch.setattr(sys, "path", [p for p in sys.path
                                      if Path(p or ".").resolve() != REPO_ROOT])
    yield
    for k in [k for k in sys.modules if k.startswith("_hermes_user_memory")]:
        del sys.modules[k]


def _check(plugin_dir, name):
    _load_like_hermes(plugin_dir, name)
    for sub in SUBMODULES:
        m = importlib.import_module(f"{name}.{sub}")
        assert m.__name__ == f"{name}.{sub}"
    backup = sys.modules[f"{name}.backup"]
    assert backup.resolve_database_url is sys.modules[f"{name}.health"].resolve_database_url


def test_loads_as_synthetic_package(isolated):
    # parent namespace package, as hermes creates for user plugins
    import types
    parent = types.ModuleType("_hermes_user_memory")
    parent.__path__ = []
    sys.modules["_hermes_user_memory"] = parent
    _check(REPO_ROOT, "_hermes_user_memory.prospecta")


def test_loads_via_profile_symlink(isolated, tmp_path):
    import types
    plugins = tmp_path / "profile" / "plugins"
    plugins.mkdir(parents=True)
    link = plugins / "prospecta"
    link.symlink_to(REPO_ROOT, target_is_directory=True)
    parent = types.ModuleType("_hermes_user_memory")
    parent.__path__ = []
    sys.modules["_hermes_user_memory"] = parent
    _check(link, "_hermes_user_memory.prospecta")


def test_top_level_still_works(monkeypatch):
    """Installed py-modules / console scripts import these as top-level."""
    for sub in ("health", "backup"):
        monkeypatch.delitem(sys.modules, sub, raising=False)
    assert importlib.import_module("backup").resolve_database_url
