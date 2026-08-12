"""The hand-computed examples in the docstrings are executable specifications."""

from __future__ import annotations

import doctest

import pytest

import devig
from analysis import kelly
from pmcal import backfill_io, fees, gamekeys, matching, odds, stats, util

MODULES = [devig, odds, fees, matching, util, stats, kelly, gamekeys, backfill_io]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_docstring_examples(module):
    result = doctest.testmod(module, verbose=False, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0, f"{module.__name__} has no doctests"
    assert result.failed == 0
