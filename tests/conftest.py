"""
conftest.py — Shared fixtures for VeriMem tests.

HOME is redirected to a temp directory before chromadb / huggingface use
so tests never write caches into the real user profile.
"""

import os
import shutil
import tempfile

_session_tmp = tempfile.mkdtemp(prefix="verimem_session_")
_original_env = {}

for _var in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH"):
    _original_env[_var] = os.environ.get(_var)

os.environ["HOME"] = _session_tmp
os.environ["USERPROFILE"] = _session_tmp
os.environ["HOMEDRIVE"] = os.path.splitdrive(_session_tmp)[0] or "C:"
os.environ["HOMEPATH"] = os.path.splitdrive(_session_tmp)[1] or _session_tmp

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _isolate_home():
    yield
    for var, orig in _original_env.items():
        if orig is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = orig
    shutil.rmtree(_session_tmp, ignore_errors=True)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="verimem_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store_path(tmp_dir):
    """Path to a temp on-disk store directory (revision / NLI tests)."""
    p = os.path.join(tmp_dir, "store")
    os.makedirs(p)
    return p
