"""
threshold_detector/coincidence.py

Convenience wrappers from binary series to ECA results, backed by the
vendored engine (:mod:`eca_analysis.eca`). Results are
:class:`eca_analysis.ECAResult` objects, which carry the coincidence
*indices* needed for the paper's compound classification.

Convention (CoinCalc / Donges, enforced by the engine)
------------------------------------------------------
* precursor: an A-event at step ``s`` is a precursor coincidence if any
  B-event lies in ``[s - tau - delT, s - tau]``; ``rate_prec = k_prec / N_A``.
* trigger: a B-event at step ``s`` is a trigger coincidence if any A-event
  lies in ``[s + tau, s + tau + delT]``; ``rate_trigg = k_trigg / N_B``.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eca_analysis import (
    ECAResult,
    eca_ts,
    eca_blocked,
    trigger_null_pmf,
    null_band,
)
from .detector import _resolve_blocks, _check_linkage


def run_eca(seriesA, seriesB, delT=4, tau=1, blocks=None, years=None,
            months=None, season_start=6, season_length=4,
            sigtest="poisson", null_model="pooled"):
    """Run Event Coincidence Analysis on two binary timeseries.

    For self-clustering (the paper's use) pass the same series twice.

    Parameters
    ----------
    seriesA, seriesB : array-like
        Binary event series (CoinCalc roles: B-events trigger A-events; see
        module docstring).
    delT : int, optional
        Coincidence window (paper default 4).
    tau : int, optional
        Minimum lag, must be >= 1 (paper default 1; avoids same-day
        self-coincidence).
    blocks / years / months / season_start / season_length :
        Season blocking, as in
        :func:`threshold_detector.detect_compound_events`. One of ``blocks``
        or ``years`` is REQUIRED so coincidences never bridge a season
        boundary. Use ``blocks='contiguous'`` explicitly for an unblocked
        single series.
    sigtest : {'poisson', 'shuffle.surrogate', None}
        Significance test (engine semantics).
    null_model : {'pooled', 'blocked'}
        'pooled' = paper Eq. (1), one binomial on the pooled totals (observed
        counts remain block-clipped); 'blocked' = exact per-season
        Poisson-binomial.

    Returns
    -------
    eca_analysis.ECAResult
        With ``prec_indices`` / ``trigg_indices`` populated (positions in the
        original series). ``compound = union(prec_indices, trigg_indices)``
        for self-ECA.
    """
    delT, tau = _check_linkage(delT, tau)
    a = np.asarray(seriesA, dtype=np.uint8)
    b = np.asarray(seriesB, dtype=np.uint8)
    blks = _resolve_blocks(len(a), blocks, years, months, season_start,
                           season_length, "run_eca")
    if len(blks) == 1 and len(blks[0]) == len(a):
        # single contiguous block: plain CoinCalc eca_ts
        return eca_ts(a, b, delT=delT, tau=tau,
                      sigtest=sigtest)
    return eca_blocked(a, b, blks, delT=delT,
                       tau=tau, sigtest=sigtest, null_model=null_model)


def run_eca_rolling(series_list, delT=4, tau=1, blocks_list=None,
                    sigtest="poisson", null_model="pooled"):
    """Run self-ECA across a list of periods (e.g. rolling 30-year windows).

    Parameters
    ----------
    series_list : list of array-like
        One binary series per period.
    blocks_list : list, optional
        Per-period blocking (same options as ``blocks`` in :func:`run_eca`).
        If omitted, each period runs as one contiguous block -- only correct
        if the period really is contiguous (not season-extracted!).

    Returns
    -------
    list of eca_analysis.ECAResult, one per period.
    """
    if blocks_list is None:
        blocks_list = ["contiguous"] * len(series_list)
    return [run_eca(s, s, delT=delT, tau=tau, blocks=b, sigtest=sigtest,
                    null_model=null_model)
            for s, b in zip(series_list, blocks_list)]


def eca_null_band(result: ECAResult, series_length=None, block_lengths=None,
                  delT=4, tau=1, null_model="pooled", lo=0.025, hi=0.975):
    """2.5/97.5% null envelope for the trigger-coincidence count of an ECA
    result (the blue band of the paper's Fig. 4), with K correctly indexed
    from 0.

    Provide ``block_lengths`` (list of per-block series lengths) for a
    blocked result, or ``series_length`` for a contiguous one.
    """
    if result.per_block is not None:
        if block_lengths is None:
            raise ValueError("blocked result: pass block_lengths")
        stats = [(t, r.n_a, r.n_b)
                 for t, r in zip(block_lengths, result.per_block)]
    else:
        if series_length is None:
            raise ValueError("contiguous result: pass series_length")
        stats = [(series_length, result.n_a, result.n_b)]
    if null_model == "pooled":
        stats = [(sum(t for t, _, _ in stats),
                  sum(a for _, a, _ in stats),
                  sum(b for _, _, b in stats))]
    elif null_model != "blocked":
        raise ValueError(f"unknown null_model {null_model!r}")
    pmf = trigger_null_pmf(stats, delT=delT, tau=tau)
    return null_band(pmf, lo=lo, hi=hi)


def summary_table(result: ECAResult) -> pd.DataFrame:
    """Human-readable summary of an ECAResult (replacement for the old
    ``EventCoincidence.summary_table``)."""
    return pd.DataFrame(
        {"Value": [result.n_a, result.n_b, result.k_prec, result.k_trigg,
                   result.rate_prec, result.rate_trigg,
                   result.pvalue_prec, result.pvalue_trigg]},
        index=["N events A", "N events B", "K precursor", "K trigger",
               "precursor coincidence rate", "trigger coincidence rate",
               "p-value precursor", "p-value trigger"])
