"""Unit tests for the notebooks."""

import glob
import os
import subprocess
import tempfile

import pytest


def _exec_notebook(path):

    file_name = tempfile.NamedTemporaryFile(suffix=".ipynb").name
    args = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=1000",
        "--ExecutePreprocessor.kernel_name=python3",
        "--output",
        file_name,
        path,
    ]
    subprocess.check_call(args)


NOTEBOOKS_DIR = "cells/notebooks"
PATHS_TO_SKIP = os.path.join(NOTEBOOKS_DIR, "explore_experimental_datasets.ipynb")
paths = sorted(glob.glob(f"{NOTEBOOKS_DIR}/*.ipynb"))


@pytest.mark.parametrize("path", paths)
def test_notebook(path):
    """Test that the notebook at path runs without bug."""
    if path in PATHS_TO_SKIP:
        pytest.skip()
    _exec_notebook(path)
