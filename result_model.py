# -*- coding: utf-8 -*-
"""
Match result probabilities (home win / draw / away win) from a goals model.

How it works
------------
1. Every past match gives two data points: home goals and away goals.
2. A Poisson regression learns, for each team, an ATTACK strength and a
   DEFENCE strength, plus one HOME ADVANTAGE for the league.
   Recent matches count more (weights halve every HALF_LIFE_DAYS).
3. For an upcoming match:
       expected home goals = exp(base + home_adv + attack[home] + defence[away])
       expected away goals = exp(base +            attack[away] + defence[home])
4. Every scoreline 0-0 ... 10-10 gets a probability (Poisson, with the usual
   Dixon-Coles tweak so 0-0 / 1-1 draws aren't under-counted), and those are
   summed into home / draw / away chances.

Only results known BEFORE the match are used, so the percentages are honest.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

HALF_LIFE_DAYS = 240
MAX_GOALS = 10
RHO = -0.08          # Dixon-Coles low-score correction
REGULARISATION = 0.002


def _clean(hist):
    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"], dayfirst=True, errors="coerce", format="mixed")
    for c in ("home_goals", "away_goals"):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h = h.dropna(subset=["date", "home_goals", "away_goals"])
    h["home_team_name"] = h["home_team_name"].astype(str).str.strip()
    h["away_team_name"] = h["away_team_name"].astype(str).str.strip()
    return h.drop_duplicates(subset=["date", "home_team_name", "away_team_name"])


def fit(hist, as_of=None):
    """Fit team strengths on matches before `as_of` (default: all)."""
    h = _clean(hist)
    if as_of is not None:
        h = h[h["date"] < as_of]
    ref = h["date"].max() if as_of is None else as_of
    h = h[h["date"] >= ref - pd.Timedelta(days=3 * 365)]
    w = 0.5 ** ((ref - h["date"]).dt.days / HALF_LIFE_DAYS)

    teams = sorted(set(h["home_team_name"]) | set(h["away_team_name"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    def rows(att, deff, home_flag):
        X = np.zeros((len(att), 2 * n + 1))
        X[np.arange(len(att)), [idx[t] for t in att]] = 1
        X[np.arange(len(att)), [n + idx[t] for t in deff]] = 1
        X[:, -1] = home_flag
        return X

    X = np.vstack([rows(h["home_team_name"], h["away_team_name"], 1),
                   rows(h["away_team_name"], h["home_team_name"], 0)])
    y = np.concatenate([h["home_goals"].values, h["away_goals"].values])
    sw = np.concatenate([w.values, w.values])

    m = PoissonRegressor(alpha=REGULARISATION, max_iter=1000)
    m.fit(X, y, sample_weight=sw)
    return {"model": m, "idx": idx, "n": n}


def _tau(i, j, lh, la):
    if i == 0 and j == 0:
        return 1 - lh * la * RHO
    if i == 0 and j == 1:
        return 1 + lh * RHO
    if i == 1 and j == 0:
        return 1 + la * RHO
    if i == 1 and j == 1:
        return 1 - RHO
    return 1.0


def expected_goals(fitted, home, away):
    m, idx, n = fitted["model"], fitted["idx"], fitted["n"]

    def x(att, deff, home_flag):
        v = np.zeros(2 * n + 1)
        if att in idx:
            v[idx[att]] = 1
        if deff in idx:
            v[n + idx[deff]] = 1
        v[-1] = home_flag
        return v
    # unknown (newly promoted, no history) teams: slightly below average
    lh = float(m.predict([x(home, away, 1)])[0]) * (0.85 if home not in idx else 1) * (1.15 if away not in idx else 1)
    la = float(m.predict([x(away, home, 0)])[0]) * (0.85 if away not in idx else 1) * (1.15 if home not in idx else 1)
    return lh, la


def probabilities(lh, la):
    g = np.arange(MAX_GOALS + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    for i in (0, 1):
        for j in (0, 1):
            M[i, j] *= _tau(i, j, lh, la)
    M /= M.sum()
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


def predict_results(fixtures, hist, as_of=None):
    """fixtures: DataFrame with fixture_id, home_team_name, away_team_name."""
    fitted = fit(hist, as_of)
    out = []
    for _, f in fixtures.iterrows():
        home, away = str(f["home_team_name"]).strip(), str(f["away_team_name"]).strip()
        lh, la = expected_goals(fitted, home, away)
        ph, pd_, pa = probabilities(lh, la)
        pick = max((("Home", ph), ("Draw", pd_), ("Away", pa)), key=lambda t: t[1])[0]
        out.append({
            "fixture_id": f["fixture_id"],
            "predicted_result": pick,
            "home_win_pct": round(ph * 100, 1),
            "draw_pct": round(pd_ * 100, 1),
            "away_win_pct": round(pa * 100, 1),
            "exp_home_goals": round(lh, 2),
            "exp_away_goals": round(la, 2),
        })
    return pd.DataFrame(out)


def score_matrix(lh, la):
    """Probability of every scoreline (rows = home goals, cols = away goals)."""
    g = np.arange(MAX_GOALS + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    for i in (0, 1):
        for j in (0, 1):
            M[i, j] *= _tau(i, j, lh, la)
    return M / M.sum()


def goal_markets(lh, la):
    """(P over 2.5 goals, P both teams score) from the same score matrix."""
    M = score_matrix(lh, la)
    i, j = np.indices(M.shape)
    return float(M[(i + j) >= 3].sum()), float(M[(i > 0) & (j > 0)].sum())
