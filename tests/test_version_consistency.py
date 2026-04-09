import re
from pathlib import Path

from verimem import __version__


def _expected_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    assert match is not None, "Could not find project version in pyproject.toml"
    return match.group(1)


def test_package_version_matches_pyproject():
    assert __version__ == _expected_version()
