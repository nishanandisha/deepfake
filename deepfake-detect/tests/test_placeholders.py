"""One import-smoke-test per module so CI has something to run for every
package before real logic lands in later stages."""

import importlib

import pytest

MODULES = [
    "src.preprocessing",
    "src.models.visual",
    "src.models.acoustic",
    "src.models.fusion",
    "src.training",
    "src.evaluation",
    "src.explain",
    "src.inference",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str):
    importlib.import_module(module_name)
