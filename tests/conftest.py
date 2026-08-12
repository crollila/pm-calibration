from __future__ import annotations

import pytest

from pmcal.panel import build_panel
from tests.factories import make_snapshots


@pytest.fixture(scope="session")
def synthetic():
    snapshots, labels = make_snapshots()
    return snapshots, labels


@pytest.fixture(scope="session")
def panel(synthetic):
    snapshots, labels = synthetic
    return build_panel(snapshots, labels)
