"""The hand-computed examples in the docstrings are executable specifications."""

from __future__ import annotations

import doctest

import pytest

import devig
from analysis import kelly
from live import analytics as live_analytics
from live import ingest as live_ingest
from live import ledger as live_ledger
from live import reference as live_reference
from live import render as live_render
from pmcal import backfill_io, fees, gamekeys, matching, odds, stats, util

MODULES = [
    devig, odds, fees, matching, util, stats, kelly, gamekeys, backfill_io,
    live_ledger, live_analytics, live_ingest, live_reference, live_render,
]  # fmt: skip


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_docstring_examples(module):
    result = doctest.testmod(module, verbose=False, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0, f"{module.__name__} has no doctests"
    assert result.failed == 0
