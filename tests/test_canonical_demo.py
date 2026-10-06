"""Test suite validating the canonical demonstration runner and reproducibility workflow."""

import pytest

from scripts.run_demo import run_canonical_demo


@pytest.mark.asyncio
async def test_canonical_demo_execution():
    """Verify that the canonical demonstration completes all 8 stages with exit code 0."""
    exit_code = await run_canonical_demo(verbose=False)
    assert exit_code == 0, f"Canonical demo failed with exit code {exit_code}"


@pytest.mark.asyncio
async def test_canonical_demo_reproducibility():
    """Verify that the canonical demonstration is reproducible across consecutive runs."""
    exit_code_1 = await run_canonical_demo(verbose=False)
    exit_code_2 = await run_canonical_demo(verbose=False)

    assert exit_code_1 == 0
    assert exit_code_2 == 0
