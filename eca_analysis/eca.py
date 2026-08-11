"""
Pure-Python Event Coincidence Analysis (ECA).

This module is a faithful port of CoinCalc's ``CC.eca.ts`` (Siegmund et al.,
CoinCalc v2.0, R), replacing the rpy2/R dependency. It additionally returns
event counts and coincidence *indices* (as the project's modified
``CoinCalc_my_version.R`` did), and supports season-restricted, year-blocked
analysis in which coincidence windows can never bridge two blocks (e.g. no
September -> next-June coincidences for JJAS analysis).

Semantics verified against the CoinCalc R source
------------------------------------------------
Precursor (sym=False): an A-event at step ``s`` is a precursor coincidence if
    any B-event lies in the inclusive window ``[s - tau - delT, s - tau]``
    (window length ``delT + 1`` steps), clipped to the series (or block) edges.
Trigger (sym=False): a B-event at step ``s`` is a trigger coincidence if any
    A-event lies in ``[s + tau, s + tau + delT]``.
sym=True widens both windows to ``[centre - delT, centre + delT]``.

Analytic ("poisson") significance test, exactly as in CoinCalc:
    p1   = 1 - (1 - (delT + 1) / (Tlen - tau)) ** N_B          (sym=False)
    p    = P(Binomial(N_A, p1) >= K_prec)                       (upper tail,
                                                                 K inclusive)
and symmetrically for the trigger rate. For sym=True CoinCalc uses
``(2*delT + 1) / Tlen``. Note: the PyPI ``event-analysis`` package uses
``delT / (T - tau)`` here instead of ``(delT + 1) / (Tlen - tau)``, so its
p-values do NOT match CoinCalc; its coincidence *rates* do (same window).

Blocked analysis
----------------
For year-blocked series the aggregate null distribution of the total
coincidence count is a sum of independent per-block binomials
(a Poisson-binomial mixture); we evaluate its upper tail exactly by
convolving the per-block binomial PMFs. The shuffle-surrogate test permutes
event positions *within each block*, which respects the block structure by
construction.

Author's-assumption flags (raise with maintainer if wrong):
* "precursor indices" are taken to be the series positions of A-events that
  are precursor coincidences, 1-based in the original R output. The custom
  ``CoinCalc_my_version.R`` was not available to verify; this is inferred
  from how the original driver consumed them (``np.array(...) - 1`` used
  directly as day indices into the cube).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.stats import binom

__all__ = [
    "ECAResult",
    "eca_ts",
    "eca_blocked",
    "flag_wet_events",
    "jjas_year_blocks",
    "trigger_null_pmf",
    "null_band",
]

SEASON_JJAS = (6, 7, 8, 9)


# --------------------------------------------------------------------------- #
# Event detection
# --------------------------------------------------------------------------- #
def flag_wet_events(timeseries: np.ndarray, N: int, wet_threshold: float) -> np.ndarray:
    """Flag wet events: runs of ``N`` consecutive steps all *exceeding*
    ``wet_threshold`` (strict ``>``). The last day of each qualifying window
    is flagged with 1, and the scan then jumps past the window so overlapping
    runs are not double-flagged (same behaviour as the original code).

    For ``N == 1`` this reduces to ``(timeseries > wet_threshold)``.

    Note: the original implementation's comment said "below the dry
    threshold"; the code (and this port) tests ``> wet_threshold``.
    """
    timeseries = np.asarray(timeseries)
    events = np.zeros(len(timeseries), dtype=int)
    if N == 1:  # fast path, bit-identical to the loop
        events[timeseries > wet_threshold] = 1
        return events
    i = 0
    while i <= len(timeseries) - N:
        if np.all(timeseries[i : i + N] > wet_threshold):
            events[i + N - 1] = 1
            i += N
        else:
            i += 1
    return events


# --------------------------------------------------------------------------- #
# Core ECA (single contiguous series) -- CoinCalc CC.eca.ts port
# --------------------------------------------------------------------------- #
@dataclass
class ECAResult:
    """Results of an event coincidence analysis.

    Rates and p-values follow CoinCalc naming; ``*_indices`` are 0-based
    positions in the input series.
    """

    n_a: int                      # 'N wet coincidences' in the original driver
    n_b: int                      # 'N dry coincidences'
    k_prec: int                   # 'N precursor'
    k_trigg: int                  # 'N trigger'
    rate_prec: float              # 'precursor coincidence rate'  K_prec / N_A
    rate_trigg: float             # 'trigger coincidence rate'    K_trigg / N_B
    prec_indices: np.ndarray      # A-event positions that are precursor coincidences
    trigg_indices: np.ndarray     # B-event positions that are trigger coincidences
    pvalue_prec: float = np.nan
    pvalue_trigg: float = np.nan
    sigtest: str = "poisson"
    # per-block diagnostics (blocked analysis only)
    per_block: Optional["list[ECAResult]"] = field(default=None, repr=False)

    def null_hypothesis(self, alpha: float = 0.05) -> tuple[bool, bool]:
        """CoinCalc 'NH precursor'/'NH trigger': True if p >= alpha
        (null NOT rejected)."""
        return bool(self.pvalue_prec >= alpha), bool(self.pvalue_trigg >= alpha)


def _coincidence_scan(
    pos_ref: np.ndarray,
    cum_other: np.ndarray,
    lo_off: int,
    hi_off: int,
    n_steps: int,
) -> np.ndarray:
    """Positions in ``pos_ref`` having >=1 event of the other series inside
    the inclusive window ``[p + lo_off, p + hi_off]``, clipped to
    ``[0, n_steps - 1]``; windows entirely outside are skipped (CoinCalc
    clipping rules). ``cum_other`` is the 0-prepended cumulative sum of the
    other binary series."""
    if len(pos_ref) == 0:
        return np.empty(0, dtype=int)
    start = pos_ref + lo_off
    end = pos_ref + hi_off
    valid = (end >= 0) & (start <= n_steps - 1)
    start = np.clip(start, 0, n_steps - 1)
    end = np.clip(end, 0, n_steps - 1)
    has = np.zeros(len(pos_ref), dtype=bool)
    has[valid] = (cum_other[end[valid] + 1] - cum_other[start[valid]]) > 0
    return pos_ref[has]


def eca_ts(
    series_a: Sequence[int],
    series_b: Sequence[int],
    delT: int = 0,
    tau: int = 0,
    sym: bool = False,
    sigtest: Optional[str] = "poisson",
    reps: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> ECAResult:
    """Event coincidence analysis of two equal-length binary series.

    Parameters mirror ``CC.eca.ts(seriesA, seriesB, delT, sym, tau, sigtest,
    reps)``. ``sigtest`` in {"poisson", "shuffle.surrogate", None}.
    """
    a = np.asarray(series_a, dtype=int)
    b = np.asarray(series_b, dtype=int)
    if a.shape != b.shape:
        raise ValueError("series A and B must have the same length")
    if delT < 0 or tau < 0:
        raise ValueError("delT and tau must be non-negative")
    if not set(np.unique(a)) <= {0, 1} or not set(np.unique(b)) <= {0, 1}:
        raise ValueError("series must be binary (0/1); binarize first")

    res = _eca_counts(a, b, delT, tau, sym)
    if sigtest == "poisson":
        res.pvalue_prec, res.pvalue_trigg = _poisson_pvalues(
            len(a), res.n_a, res.n_b, res.k_prec, res.k_trigg, delT, tau, sym
        )
    elif sigtest == "shuffle.surrogate":
        res.pvalue_prec, res.pvalue_trigg = _shuffle_pvalues(
            [len(a)], [(res.n_a, res.n_b)], res.rate_prec, res.rate_trigg,
            delT, tau, sym, reps, rng,
        )
        res.sigtest = "shuffle.surrogate"
    elif sigtest is not None:
        raise ValueError(f"unknown sigtest {sigtest!r}")
    return res


def _eca_counts(a, b, delT, tau, sym) -> ECAResult:
    n_steps = len(a)
    pos_a = np.flatnonzero(a == 1)
    pos_b = np.flatnonzero(b == 1)
    cum_a = np.concatenate(([0], np.cumsum(a)))
    cum_b = np.concatenate(([0], np.cumsum(b)))

    # precursor: B in [s - tau - delT, s - tau(+delT if sym)] around A-events
    prec_hi = -tau + (delT if sym else 0)
    prec_idx = _coincidence_scan(pos_a, cum_b, -tau - delT, prec_hi, n_steps)
    # trigger: A in [s + tau(-delT if sym), s + tau + delT] around B-events
    trig_lo = tau - (delT if sym else 0)
    trig_idx = _coincidence_scan(pos_b, cum_a, trig_lo, tau + delT, n_steps)

    n_a, n_b = len(pos_a), len(pos_b)
    return ECAResult(
        n_a=n_a,
        n_b=n_b,
        k_prec=len(prec_idx),
        k_trigg=len(trig_idx),
        rate_prec=len(prec_idx) / n_a if n_a else np.nan,
        rate_trigg=len(trig_idx) / n_b if n_b else np.nan,
        prec_indices=prec_idx,
        trigg_indices=trig_idx,
    )


def _block_success_probs(t_len: int, n_a: int, n_b: int, delT: int, tau: int, sym: bool):
    """CoinCalc per-event coincidence probabilities under the null."""
    if sym:
        w = (2 * delT + 1) / t_len
        p_prec = 1.0 - (1.0 - w) ** n_b
        p_trig = 1.0 - (1.0 - w) ** n_a
    else:
        w = (delT + 1) / (t_len - tau)
        p_prec = 1.0 - (1.0 - w) ** n_b
        p_trig = 1.0 - (1.0 - w) ** n_a
    return p_prec, p_trig


def _poisson_pvalues(t_len, n_a, n_b, k_prec, k_trigg, delT, tau, sym):
    p_prec, p_trig = _block_success_probs(t_len, n_a, n_b, delT, tau, sym)
    # CoinCalc sums the binomial PMF from K inclusive: P(X >= K) = sf(K-1)
    pv_prec = binom.sf(k_prec - 1, n_a, p_prec) if n_a else np.nan
    pv_trig = binom.sf(k_trigg - 1, n_b, p_trig) if n_b else np.nan
    return float(pv_prec), float(pv_trig)


# --------------------------------------------------------------------------- #
# Blocked (season/year-grouped) ECA
# --------------------------------------------------------------------------- #
def jjas_year_blocks(
    years: Sequence[int],
    months: Sequence[int],
    season: Sequence[int] = SEASON_JJAS,
) -> "list[np.ndarray]":
    """Return a list of index arrays, one per (year, season) block, in
    chronological order. Indices refer to the *original* series so results
    can be mapped straight back to dates."""
    years = np.asarray(years)
    months = np.asarray(months)
    in_season = np.isin(months, list(season))
    blocks = []
    for yr in np.unique(years[in_season]):
        blocks.append(np.flatnonzero(in_season & (years == yr)))
    return blocks


def eca_blocked(
    series_a: Sequence[int],
    series_b: Sequence[int],
    blocks: "list[np.ndarray]",
    delT: int = 0,
    tau: int = 0,
    sym: bool = False,
    sigtest: Optional[str] = "poisson",
    reps: int = 1000,
    rng: Optional[np.random.Generator] = None,
    null_model: str = "blocked",
) -> ECAResult:
    """ECA where each block is analysed independently: coincidence windows are
    clipped at block edges, so events can never coincide across blocks
    (e.g. September -> next June). Counts are aggregated over blocks;
    rates are (sum K) / (sum N).

    Significance of the aggregate (``sigtest="poisson"``), controlled by
    ``null_model``:
    * "blocked" (default): exact upper tail of the sum of per-block binomials
      (Poisson-binomial), each block using CoinCalc's per-block success
      probability. Reduces exactly to CoinCalc's test for a single block.
    * "pooled": CoinCalc's/Donges' single binomial (Eq. 1) evaluated on the
      pooled totals (T = sum of block lengths, N = summed events) -- the
      per-block model at window-average event density. Note the observed
      counts remain block-clipped, while this null permits cross-block
      coincidences; the mismatch is bounded by boundary clipping,
      ~(delT+tau)/block_length of trigger windows.
    ``sigtest="shuffle.surrogate"`` shuffles events *within* their block.

    ``prec_indices``/``trigg_indices`` are positions in the original
    (unblocked) series.
    """
    a = np.asarray(series_a, dtype=int)
    b = np.asarray(series_b, dtype=int)
    per_block: list[ECAResult] = []
    prec_idx_all, trig_idx_all = [], []
    for blk in blocks:
        r = eca_ts(a[blk], b[blk], delT=delT, tau=tau, sym=sym, sigtest=None)
        per_block.append(r)
        prec_idx_all.append(blk[r.prec_indices])
        trig_idx_all.append(blk[r.trigg_indices])

    n_a = sum(r.n_a for r in per_block)
    n_b = sum(r.n_b for r in per_block)
    k_prec = sum(r.k_prec for r in per_block)
    k_trigg = sum(r.k_trigg for r in per_block)
    agg = ECAResult(
        n_a=n_a,
        n_b=n_b,
        k_prec=k_prec,
        k_trigg=k_trigg,
        rate_prec=k_prec / n_a if n_a else np.nan,
        rate_trigg=k_trigg / n_b if n_b else np.nan,
        prec_indices=np.concatenate(prec_idx_all) if prec_idx_all else np.empty(0, int),
        trigg_indices=np.concatenate(trig_idx_all) if trig_idx_all else np.empty(0, int),
        per_block=per_block,
    )

    if sigtest == "poisson":
        if null_model == "pooled":
            t_tot = sum(len(blk) for blk in blocks)
            agg.pvalue_prec, agg.pvalue_trigg = _poisson_pvalues(
                t_tot, n_a, n_b, k_prec, k_trigg, delT, tau, sym)
        elif null_model == "blocked":
            agg.pvalue_prec = _poisson_binomial_tail(
                [(len(blk), r.n_a, r.n_b, r.k_prec)
                 for blk, r in zip(blocks, per_block)],
                k_prec, delT, tau, sym, direction="prec",
            )
            agg.pvalue_trigg = _poisson_binomial_tail(
                [(len(blk), r.n_a, r.n_b, r.k_trigg)
                 for blk, r in zip(blocks, per_block)],
                k_trigg, delT, tau, sym, direction="trig",
            )
        else:
            raise ValueError(f"unknown null_model {null_model!r}")
    elif sigtest == "shuffle.surrogate":
        agg.pvalue_prec, agg.pvalue_trigg = _shuffle_pvalues(
            [len(blk) for blk in blocks],
            [(r.n_a, r.n_b) for r in per_block],
            agg.rate_prec, agg.rate_trigg, delT, tau, sym, reps, rng,
        )
        agg.sigtest = "shuffle.surrogate"
    elif sigtest is not None:
        raise ValueError(f"unknown sigtest {sigtest!r}")
    return agg


def _poisson_binomial_tail(block_stats, k_obs, delT, tau, sym, direction):
    """Exact P(sum of per-block Binomials >= k_obs) via PMF convolution."""
    pmf = np.array([1.0])
    for t_len, n_a, n_b, _k in block_stats:
        p_prec, p_trig = _block_success_probs(t_len, n_a, n_b, delT, tau, sym)
        n = n_a if direction == "prec" else n_b
        p = p_prec if direction == "prec" else p_trig
        if n == 0:
            continue
        pmf = np.convolve(pmf, binom.pmf(np.arange(n + 1), n, p))
    if len(pmf) == 1:  # no events at all
        return float("nan")
    pmf = pmf / pmf.sum()  # guard tiny FP drift
    return float(pmf[int(k_obs):].sum())


def trigger_null_pmf(block_stats, delT, tau, sym=False):
    """PMF of the total trigger-coincidence count K_trigg under the null.

    ``block_stats`` is a list of ``(t_len, n_a, n_b)`` -- one tuple per block
    for the year-blocked case, or a single tuple for the concatenated case.
    Each block contributes an independent ``Binomial(n_b, p_trig)`` with
    CoinCalc's ``p_trig``; the total is their convolution (a Poisson-binomial
    for >1 block, an ordinary binomial for one). Returns a pmf array indexed
    by K = 0, 1, ..., sum(n_b). This is the same maths as the p-value tail,
    exposed so the plot band and the significance test can never disagree.
    """
    pmf = np.array([1.0])
    for t_len, n_a, n_b in block_stats:
        if n_b == 0:
            continue
        _, p_trig = _block_success_probs(t_len, n_a, n_b, delT, tau, sym)
        pmf = np.convolve(pmf, binom.pmf(np.arange(n_b + 1), n_b, p_trig))
    return pmf / pmf.sum()


def null_band(pmf, lo=0.025, hi=0.975, interpolate=True):
    """Lower/upper quantiles of a K distribution given its ``pmf``.

    With ``interpolate=True`` (default) the quantile is linearly interpolated
    on the CDF over K = 0, 1, ... (matching the smooth band in the original
    plots but with correct K indexing, i.e. K starts at 0 not 1). With
    ``interpolate=False`` it returns the smallest integer K whose CDF >= the
    level (a strict, conservative band).
    """
    k = np.arange(len(pmf))
    cdf = np.cumsum(pmf)
    if interpolate:
        return float(np.interp(lo, cdf, k)), float(np.interp(hi, cdf, k))
    return int(k[np.searchsorted(cdf, lo)]), int(k[np.searchsorted(cdf, hi)])


def _shuffle_pvalues(block_lens, block_counts, rate_prec_obs, rate_trigg_obs,
                     delT, tau, sym, reps, rng):
    """CoinCalc shuffle.surrogate test, generalised to blocks: uniformly
    re-place N_A and N_B events within each block, recompute aggregate rates
    ``reps`` times; p = fraction of surrogate rates strictly greater than the
    observed rate (CoinCalc's ``1 - ecdf(sur)(obs)``)."""
    rng = rng or np.random.default_rng()
    n_a_tot = sum(na for na, _ in block_counts)
    n_b_tot = sum(nb for _, nb in block_counts)
    if n_a_tot == 0 or n_b_tot == 0:
        return float("nan"), float("nan")
    sur_prec = np.empty(reps)
    sur_trig = np.empty(reps)
    for i in range(reps):
        k_p = k_t = 0
        for t_len, (n_a, n_b) in zip(block_lens, block_counts):
            a = np.zeros(t_len, dtype=int)
            b = np.zeros(t_len, dtype=int)
            a[rng.choice(t_len, size=n_a, replace=False)] = 1
            b[rng.choice(t_len, size=n_b, replace=False)] = 1
            r = _eca_counts(a, b, delT, tau, sym)
            k_p += r.k_prec
            k_t += r.k_trigg
        sur_prec[i] = k_p / n_a_tot
        sur_trig[i] = k_t / n_b_tot
    return (
        float(np.mean(sur_prec > rate_prec_obs)),
        float(np.mean(sur_trig > rate_trigg_obs)),
    )