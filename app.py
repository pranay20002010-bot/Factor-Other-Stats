"""
Factor Lab - Indian 4-factor decomposition + volatility + Monte Carlo (Streamlit)
================================================================================
Run:   streamlit run app.py
Data:  put the IIM-A factor CSV next to this file (or in ./data/). It is picked up
       automatically; the sidebar uploader is only an optional override.

Sections
  1. Four-factor regression (Market, SMB, HML, WML) with White / robust standard errors
  2. Hurst exponent (R/S with Anis-Lloyd correction + DFA) and integer differencing order
  3. ARIMA (BIC-selected) with an honest hold-out test against a random walk
  4. EWMA variance and best-BIC GARCH-family model, model-implied volatility
  5. Monte Carlo probability of reaching a target price
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from arch import arch_model
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter
from scipy.special import gammaln
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- config
TDAYS = 250  # trading days per year used for annualisation (NSE ~ 248-250)
FACTORS = ["MF", "SMB", "HML", "WML"]
FLABEL = {"MF": "Market (MF)", "SMB": "Size (SMB)", "HML": "Value (HML)", "WML": "Momentum (WML)"}
# categorical slots 1-4 of the reference palette (blue, orange, aqua, yellow) + neutrals
FCOL = {"MF": "#2a78d6", "SMB": "#eb6834", "HML": "#1baf7a", "WML": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LOOKBACKS = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "Max": None}
SE_TYPES = {
    "White (HC0)": dict(cov_type="HC0"),
    "White HC1 (small-sample adj.)": dict(cov_type="HC1"),
    "HC3 (best in small samples)": dict(cov_type="HC3"),
    "Newey-West HAC (5 lags)": dict(cov_type="HAC", cov_kwds={"maxlags": 5}),
}
GARCH_CANDIDATES = {
    "GARCH(1,1)": dict(vol="GARCH", p=1, o=0, q=1),
    "GARCH(1,2)": dict(vol="GARCH", p=1, o=0, q=2),
    "GARCH(2,1)": dict(vol="GARCH", p=2, o=0, q=1),
    "GARCH(2,2)": dict(vol="GARCH", p=2, o=0, q=2),
    "GJR-GARCH(1,1)": dict(vol="GARCH", p=1, o=1, q=1),
    "EGARCH(1,1)": dict(vol="EGARCH", p=1, o=1, q=1),
}


# =========================================================================== DATA
def _clean_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["MF", "SMB", "HML", "WML", "RF"]].dropna()  # values are in % per day


@st.cache_data(show_spinner=False)
def read_factor_path(path: str, mtime: float) -> pd.DataFrame:
    return _clean_factors(pd.read_csv(path))


@st.cache_data(show_spinner=False)
def read_factor_bytes(raw: bytes) -> pd.DataFrame:
    return _clean_factors(pd.read_csv(io.BytesIO(raw)))


def find_factor_file() -> Path | None:
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    for d in (here, here / "data", Path.cwd(), Path.cwd() / "data"):
        for pat in ("*FourFactors*.csv", "*four*factor*.csv", "factors.csv"):
            hits = sorted(d.glob(pat))
            if hits:
                return hits[-1]  # newest vintage if several
    return None


def resolve_symbol(raw: str, exch: str) -> str:
    s = raw.strip().upper().replace(" ", "")
    if s.endswith((".NS", ".BO")):
        return s
    if s.isdigit():  # BSE scrip code
        return s + ".BO"
    return s + (".NS" if exch == "NSE" else ".BO")


@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def load_prices(symbol: str) -> pd.Series:
    df = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No price data returned for {symbol}")
    s = df["Close"].dropna().copy()
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s[s > 0].sort_index()


def window(obj, years):
    if years is None or len(obj) == 0:
        return obj
    return obj[obj.index > obj.index.max() - pd.DateOffset(years=years)]


# ================================================================ FACTOR REGRESSION
def factor_regression(ret_pct: pd.Series, fac: pd.DataFrame, se_key: str, years):
    df = pd.concat([ret_pct.rename("R"), fac], axis=1, join="inner").dropna()
    df = window(df, years)
    if len(df) < 120:
        raise ValueError(f"Only {len(df)} overlapping days with the factor data (need >= 120).")
    y = df["R"] - df["RF"]  # excess return, % per day
    X = sm.add_constant(df[FACTORS])
    ols = sm.OLS(y, X).fit()
    rob = sm.OLS(y, X).fit(**SE_TYPES[se_key])
    ci = rob.conf_int()
    tab = pd.DataFrame({
        "Coef": rob.params,
        "Robust SE": rob.bse,
        "t (robust)": rob.tvalues,
        "p (robust)": rob.pvalues,
        "CI 2.5%": ci[0],
        "CI 97.5%": ci[1],
        "Classical SE": ols.bse,
        "t (classical)": ols.tvalues,
    })
    tab.index = ["Alpha (daily %)"] + [FLABEL[f] for f in FACTORS]
    resid = ols.resid
    diag = {
        "n": len(df), "start": df.index.min(), "end": df.index.max(),
        "r2": rob.rsquared, "adj_r2": rob.rsquared_adj,
        "alpha_ann": rob.params["const"] * TDAYS, "alpha_p": rob.pvalues["const"],
        "excess_kurtosis": stats.kurtosis(resid),
        "jb_p": stats.jarque_bera(resid).pvalue,
        "bp_p": het_breuschpagan(resid, X)[1],
        "dw": durbin_watson(resid),
    }
    means = df[FACTORS].mean()
    attrib = {"Alpha": rob.params["const"] * TDAYS}
    for f in FACTORS:
        attrib[FLABEL[f]] = rob.params[f] * means[f] * TDAYS
    attrib["Total excess return"] = float(y.mean() * TDAYS)
    w = int(min(250, len(df) // 2))
    roll = RollingOLS(y, X, window=w).fit(params_only=True).params.dropna()[FACTORS]
    return dict(table=tab, diag=diag, attrib=attrib, rolling=roll, betas=rob.params[FACTORS],
                factor_means_ann=means * TDAYS, se_key=se_key, roll_window=w)


def interpret_factors(res) -> list[str]:
    t = res["table"]
    out = []
    names = {FLABEL[f]: f for f in FACTORS}
    for lab, f in names.items():
        b, tv = t.loc[lab, "Coef"], t.loc[lab, "t (robust)"]
        sig = abs(tv) >= 2
        if f == "MF":
            tone = "more aggressive than the market" if b > 1.1 else "defensive vs the market" if b < 0.9 else "roughly market-like"
            out.append(f"**Market beta {b:.2f}** - {tone}.")
        elif f == "SMB":
            out.append(f"**Size {b:+.2f}** - " + (("small-cap tilt" if b > 0 else "large-cap tilt") if sig else "no significant size tilt") + ".")
        elif f == "HML":
            out.append(f"**Value {b:+.2f}** - " + (("value tilt" if b > 0 else "growth tilt") if sig else "no significant value/growth tilt") + ".")
        else:
            out.append(f"**Momentum {b:+.2f}** - " + (("trades with winners (momentum exposure)" if b > 0 else "contrarian / loser-like exposure") if sig else "no significant momentum exposure") + ".")
    a = res["diag"]
    out.append(f"**Alpha {a['alpha_ann']:+.1f}% p.a.** - " + ("statistically significant" if a["alpha_p"] < 0.05 else "not statistically different from zero") + " (robust p = %.3f)." % a["alpha_p"])
    return out


# ============================================================================ HURST
def _log_sizes(n, lo, hi_frac=2, k=20):
    hi = max(lo * 2, n // hi_frac)
    return np.unique(np.floor(np.logspace(np.log10(lo), np.log10(hi), k)).astype(int))


def _expected_rs(n: int) -> float:
    """Anis-Lloyd (1976) expected R/S of an iid series of length n."""
    i = np.arange(1, n)
    s = np.sum(np.sqrt((n - i) / i))
    return ((n - 0.5) / n) * np.exp(gammaln((n - 1) / 2) - gammaln(n / 2)) / np.sqrt(np.pi) * s


def hurst_rs(x: np.ndarray, lo: int = 10):
    x = np.asarray(x, float)
    n = len(x)
    sizes, rs, ers = [], [], []
    for s in _log_sizes(n, lo):
        m = n // s
        if m < 1:
            continue
        ch = x[: m * s].reshape(m, s)
        z = np.cumsum(ch - ch.mean(axis=1, keepdims=True), axis=1)
        R = z.max(axis=1) - z.min(axis=1)
        S = ch.std(axis=1, ddof=1)
        ok = S > 0
        if ok.sum() == 0:
            continue
        sizes.append(s)
        rs.append(np.mean(R[ok] / S[ok]))
        ers.append(_expected_rs(s))
    ls, lrs, lers = np.log(sizes), np.log(rs), np.log(ers)
    h_raw = np.polyfit(ls, lrs, 1)[0]
    h_corr = 0.5 + np.polyfit(ls, lrs - lers, 1)[0]  # Anis-Lloyd-Peters correction
    return h_corr, h_raw, ls, lrs, lers


def hurst_dfa(x: np.ndarray, lo: int = 10):
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    n = len(y)
    sizes, F = [], []
    for s in _log_sizes(n, lo, hi_frac=4):
        m = n // s
        if m < 2:
            continue
        seg = y[: m * s].reshape(m, s)
        t = np.arange(s)
        A = np.column_stack([t, np.ones(s)])
        coef = np.linalg.lstsq(A, seg.T, rcond=None)[0]
        resid = seg - (A @ coef).T
        sizes.append(s)
        F.append(np.sqrt(np.mean(resid ** 2)))
    return np.polyfit(np.log(sizes), np.log(F), 1)[0]


@st.cache_data(show_spinner="Hurst exponent + permutation test...")
def hurst_analysis(logret: pd.Series, n_perm: int = 300, seed: int = 7):
    x = logret.values
    h, h_raw, ls, lrs, lers = hurst_rs(x)
    dfa = hurst_dfa(x)
    rng = np.random.default_rng(seed)
    null = np.array([hurst_rs(rng.permutation(x))[0] for _ in range(n_perm)])
    p = (np.sum(np.abs(null - 0.5) >= abs(h - 0.5)) + 1) / (n_perm + 1)
    return dict(h_rs=h, h_rs_raw=h_raw, h_dfa=dfa, h_vol=hurst_dfa(np.abs(x)), p=p,
                null_lo=float(np.percentile(null, 2.5)), null_hi=float(np.percentile(null, 97.5)),
                ls=ls, lrs=lrs, lers=lers, n=len(x))


# ===================================================================== STATIONARITY
@st.cache_data(show_spinner="Stationarity tests...")
def stationarity_analysis(logp: pd.Series):
    x = logp.values.astype(float)
    rows, chosen, chosen_adf = [], None, None
    for d in range(3):
        s = np.diff(x, n=d) if d else x
        a = adfuller(s, regression="c", autolag="AIC")
        k = kpss(s, regression="c", nlags="auto")
        adf_ok, kpss_ok = a[1] < 0.05, k[1] > 0.05
        rows.append({"d": d, "Series": ["log price", "1st difference (log return)", "2nd difference"][d],
                     "ADF stat": a[0], "ADF p": a[1], "KPSS stat": k[0], "KPSS p": k[1],
                     "ADF: stationary?": "yes" if adf_ok else "no",
                     "KPSS: stationary?": "yes" if kpss_ok else "no"})
        if chosen_adf is None and adf_ok:
            chosen_adf = d
        if chosen is None and adf_ok and kpss_ok:
            chosen = d
    d_final = chosen if chosen is not None else (chosen_adf if chosen_adf is not None else 1)
    return pd.DataFrame(rows), d_final, chosen, chosen_adf


# ======================================================================== ARIMA
@st.cache_data(show_spinner="Fitting ARIMA grid...")
def arima_analysis(logp: pd.Series, d: int, horizon: int = 10, max_pq: int = 3):
    x = logp.values.astype(float)
    trend = {0: "c", 1: "t", 2: "n"}[d]
    rows = []
    for p in range(max_pq + 1):
        for q in range(max_pq + 1):
            try:
                m = ARIMA(x, order=(p, d, q), trend=trend).fit()
                rows.append(dict(p=p, d=d, q=q, AIC=m.aic, BIC=m.bic))
            except Exception:
                continue
    grid = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
    p, _, q = (int(grid.loc[0, c]) for c in ("p", "d", "q"))
    res = ARIMA(x, order=(p, d, q), trend=trend).fit()
    fc = res.get_forecast(horizon)
    ci = fc.conf_int(alpha=0.05)
    last = x[-1]
    # hold-out: fit on train, one-step-ahead forecasts on the last k days (parameters frozen)
    k = int(min(250, len(x) // 4))
    n_tr = len(x) - k
    tr = ARIMA(x[:n_tr], order=(p, d, q), trend=trend).fit()
    full = tr.apply(x)
    pred = np.asarray(full.get_prediction(start=n_tr, end=len(x) - 1, dynamic=False).predicted_mean)
    prev = x[n_tr - 1: len(x) - 1]
    f_ret, a_ret = pred - prev, x[n_tr:] - prev
    mu_tr = np.mean(np.diff(x[:n_tr]))
    rm = lambda e: float(np.sqrt(np.mean(e ** 2)))
    rmse_arima, rmse_zero, rmse_drift = rm(a_ret - f_ret), rm(a_ret), rm(a_ret - mu_tr)
    hits = int(np.sum(np.sign(f_ret) == np.sign(a_ret)))
    hit_p = stats.binomtest(hits, k, 0.5).pvalue
    lb = acorr_ljungbox(res.resid[1:] if d else res.resid, lags=[10], return_df=True)
    arch_p = het_arch(res.resid[1:] if d else res.resid, nlags=5)[1]
    return dict(
        grid=grid.head(6), order=(p, d, q), trend=trend, bic=res.bic,
        params=pd.Series(res.params, index=res.param_names), pvalues=pd.Series(res.pvalues, index=res.param_names),
        fc_price=np.exp(np.asarray(fc.predicted_mean)), fc_lo=np.exp(ci[:, 0]), fc_hi=np.exp(ci[:, 1]),
        last_price=float(np.exp(last)), k=k, rmse_arima=rmse_arima, rmse_zero=rmse_zero, rmse_drift=rmse_drift,
        hit_rate=hits / k, hit_p=hit_p, up_rate=float(np.mean(a_ret > 0)),
        lb_p=float(lb["lb_pvalue"].iloc[0]), arch_p=float(arch_p),
    )


# ================================================================== VOLATILITY
def ewma_conditional_var(r: np.ndarray, lam: float) -> np.ndarray:
    r2 = np.asarray(r, float) ** 2
    v0 = float(np.mean(r2[:30]))
    s = lfilter([1 - lam], [1, -lam], r2, zi=[lam * v0])[0]  # s_t = lam*s_{t-1} + (1-lam) r_t^2
    return np.concatenate([[v0], s[:-1]]), float(s[-1])  # cond. var at t (known t-1), forecast for t+1


def ewma_lambda_mle(r: np.ndarray) -> float:
    def nll(lam):
        v, _ = ewma_conditional_var(r, lam)
        return 0.5 * np.sum(np.log(v) + np.asarray(r) ** 2 / v)
    return float(minimize_scalar(nll, bounds=(0.80, 0.995), method="bounded").x)


@st.cache_data(show_spinner="Selecting GARCH specification...")
def garch_selection(logret_pct: pd.Series):
    rows, best, best_bic = [], None, np.inf
    for name, spec in GARCH_CANDIDATES.items():
        try:
            r = arch_model(logret_pct, mean="Constant", dist="t", rescale=False, **spec).fit(disp="off", options={"maxiter": 800})
            ok = r.convergence_flag == 0
            rows.append({"Model": name, "BIC": r.bic, "AIC": r.aic, "LogLik": r.loglikelihood, "Converged": ok})
            if ok and r.bic < best_bic:
                best, best_bic = name, r.bic
        except Exception as e:  # pragma: no cover
            rows.append({"Model": name, "BIC": np.nan, "AIC": np.nan, "LogLik": np.nan, "Converged": False})
    if best is None:
        raise RuntimeError("No GARCH specification converged.")
    return pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True), best


@st.cache_resource(show_spinner=False)
def fit_garch(logret_pct: pd.Series, name: str):
    return arch_model(logret_pct, mean="Constant", dist="t", rescale=False, **GARCH_CANDIDATES[name]).fit(disp="off", options={"maxiter": 800})


def _std_rng(nu, seed):
    g = np.random.default_rng(seed)
    sc = np.sqrt((nu - 2) / nu)
    return lambda size: g.standard_t(nu, size) * sc


def garch_sim(res, horizon: int, sims: int, seed: int):
    """Returns (residuals %, conditional variances %^2), each (sims, horizon)."""
    nu = float(res.params["nu"])
    f = res.forecast(horizon=horizon, method="simulation", simulations=sims, reindex=False, rng=_std_rng(nu, seed))
    return f.simulations.residuals[-1], f.simulations.variances[-1]


def garch_summary(res, name: str):
    p = res.params
    keys = list(p.index)
    if "EGARCH" in name:
        pers = float(sum(p[k] for k in keys if k.startswith("beta")))
    else:
        pers = float(sum(p[k] for k in keys if k.startswith("alpha") or k.startswith("beta")) + 0.5 * sum(p[k] for k in keys if k.startswith("gamma")))
    half = float(np.log(0.5) / np.log(pers)) if 0 < pers < 1 else np.inf
    _, var = garch_sim(res, 250, 2000, seed=11)
    mv = var.mean(axis=0)  # expected daily variance (%^2) per horizon step
    return dict(persistence=pers, half_life=half, nu=float(p["nu"]), exp_var=mv,
                params=pd.DataFrame({"Coef": p, "Std err": res.std_err, "p": res.pvalues}),
                lb_p=float(acorr_ljungbox((res.resid / res.conditional_volatility).dropna() ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]),
                cond_vol_ann=res.conditional_volatility * np.sqrt(TDAYS))


def avg_vol(exp_var_pct2: np.ndarray, h: int) -> float:
    h = min(h, len(exp_var_pct2))
    return float(np.sqrt(exp_var_pct2[:h].mean()) / 100 * np.sqrt(TDAYS))


# ============================================================== MONTE CARLO
def mc_run(S0, target, H, N, drift_ann, mode, sig_ann=None, nu_const=None, garch_res=None, seed=1, batch=5000):
    """Simulate N paths of daily closes. Returns dict of results.
    mode: 'const' (constant vol; normal if nu_const is None else Student-t) or 'garch' (path-dependent)."""
    up = target >= S0
    paths = np.empty((N, H), dtype=np.float32)
    mu_log_d = np.log1p(drift_ann) / TDAYS
    done, b = 0, 0
    while done < N:
        n = min(batch, N - done)
        if mode == "const":
            g = np.random.default_rng(seed + b)
            half = (n + 1) // 2
            z = g.standard_normal((half, H)) if nu_const is None else g.standard_t(nu_const, (half, H)) * np.sqrt((nu_const - 2) / nu_const)
            z = np.vstack([z, -z])[:n]  # antithetic variates
            sd = sig_ann / np.sqrt(TDAYS)
            inc = mu_log_d - 0.5 * sd ** 2 + sd * z
        else:
            resid, var = garch_sim(garch_res, H, n, seed + b)
            inc = mu_log_d - 0.5 * var / 1e4 + resid / 100.0
        paths[done:done + n] = S0 * np.exp(np.cumsum(inc, axis=1))
        done += n
        b += 1
    term = paths[:, -1].astype(float)
    hit_mat = paths >= target if up else paths <= target
    touched = hit_mat.any(axis=1)
    first = np.where(touched, hit_mat.argmax(axis=1) + 1, 0)
    p_term = float(np.mean(term >= target if up else term <= target))
    p_touch = float(touched.mean())
    se = lambda p: float(np.sqrt(max(p * (1 - p), 1e-12) / N))
    qs = [5, 25, 50, 75, 95]
    return dict(
        up=up, p_term=p_term, p_touch=p_touch, se_term=se(p_term), se_touch=se(p_touch),
        term=term, fan={q: np.percentile(paths, q, axis=0) for q in qs},
        quantiles={q: float(np.percentile(term, q)) for q in qs}, mean_term=float(term.mean()),
        median_hit_day=float(np.median(first[touched])) if touched.any() else np.nan,
        cum_hit=np.array([np.mean((first > 0) & (first <= t)) for t in range(1, H + 1)]),
    )


def lognormal_term_prob(S0, K, H, drift_ann, sig_ann):
    s = sig_ann * np.sqrt(H / TDAYS)
    m = (np.log1p(drift_ann) - 0.5 * sig_ann ** 2) * H / TDAYS
    z = (np.log(K / S0) - m) / s
    return float(1 - stats.norm.cdf(z)) if K >= S0 else float(stats.norm.cdf(z))


# ================================================================== PLOT HELPERS
def style(fig, title=None, height=380, legend=True):
    fig.update_layout(
        template="plotly_white", height=height, title=dict(text=title, font=dict(size=15, color=INK)) if title else None,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10), showlegend=legend,
        legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified", font=dict(color=MUTED),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ======================================================================= SECTIONS
def section_factors(res, symbol):
    d = res["diag"]
    c = st.columns(5)
    c[0].metric("R²", f"{d['r2']:.3f}", help="Share of daily return variance explained by the four factors")
    c[1].metric("Adj. R²", f"{d['adj_r2']:.3f}")
    c[2].metric("Alpha (p.a.)", f"{d['alpha_ann']:+.1f}%", help="Intercept x trading days; excess return not explained by the factors")
    c[3].metric("Observations", f"{d['n']:,}")
    c[4].metric("Window", f"{d['start']:%b %Y} - {d['end']:%b %Y}")
    st.caption(f"Standard errors: **{res['se_key']}**. Dependent variable = stock return minus RF (daily %).")

    t = res["table"].copy()
    t["Sig."] = np.where(t["p (robust)"] < 0.01, "***", np.where(t["p (robust)"] < 0.05, "**", np.where(t["p (robust)"] < 0.10, "*", "")))
    st.dataframe(t.round(4), width="stretch")

    left, right = st.columns(2)
    b, tb = res["betas"], res["table"]
    fig = go.Figure()
    for f in FACTORS:
        row = tb.loc[FLABEL[f]]
        fig.add_trace(go.Bar(x=[FLABEL[f]], y=[row["Coef"]], marker_color=FCOL[f], name=FLABEL[f],
                             error_y=dict(type="data", symmetric=False, array=[row["CI 97.5%"] - row["Coef"]], arrayminus=[row["Coef"] - row["CI 2.5%"]], color=MUTED),
                             hovertemplate="%{x}<br>beta %{y:.3f}<extra></extra>"))
    style(fig, "Factor loadings (95% robust CI)", legend=False).update_layout(hovermode="closest")
    left.plotly_chart(fig, width="stretch")

    a = res["attrib"]
    names, vals = list(a.keys()), list(a.values())
    cols = [MUTED] + [FCOL[f] for f in FACTORS] + [INK]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=cols, hovertemplate="%{x}<br>%{y:.2f}% p.a.<extra></extra>"))
    style(fig, "Where the excess return came from (% p.a.)", legend=False).update_layout(hovermode="closest")
    right.plotly_chart(fig, width="stretch")
    st.caption("Attribution is exact: average excess return = alpha + sum(beta x average factor return).")

    roll = res["rolling"]
    fig = go.Figure()
    for f in FACTORS:
        fig.add_trace(go.Scatter(x=roll.index, y=roll[f], name=FLABEL[f], line=dict(color=FCOL[f], width=2)))
    style(fig, f"Rolling {res['roll_window']}-day loadings", height=340)
    st.plotly_chart(fig, width="stretch")

    with st.expander("Diagnostics - why robust standard errors"):
        st.write(f"- Residual excess kurtosis: **{d['excess_kurtosis']:.1f}** (normal = 0); Jarque-Bera p = {d['jb_p']:.3g}")
        st.write(f"- Breusch-Pagan heteroskedasticity p = **{d['bp_p']:.3g}** " + ("(heteroskedastic - robust SEs matter)" if d["bp_p"] < 0.05 else "(no strong evidence)"))
        st.write(f"- Durbin-Watson = {d['dw']:.2f} (2 = no autocorrelation). If far from 2, switch to Newey-West.")
        cmp = res["table"][["Robust SE", "Classical SE"]].copy()
        cmp["Robust / Classical"] = cmp["Robust SE"] / cmp["Classical SE"]
        st.dataframe(cmp.round(4), width="stretch")
        st.caption("White SEs correct for heteroskedasticity (volatility clustering), which is what fat-tailed returns bring. "
                   "They do not fix autocorrelation (use Newey-West) or non-synchronous trading in illiquid stocks.")
    st.markdown("**Reading the result**")
    for line in interpret_factors(res):
        st.markdown("- " + line)


def section_memory(h, stn):
    st.subheader("Hurst exponent")
    c = st.columns(4)
    c[0].metric("H (R/S, corrected)", f"{h['h_rs']:.3f}", help="Rescaled-range with Anis-Lloyd small-sample correction, on log returns")
    c[1].metric("H (DFA)", f"{h['h_dfa']:.3f}", help="Detrended fluctuation analysis - robust to non-stationarity")
    c[2].metric("Permutation p-value", f"{h['p']:.3f}", help="Chance of an H this far from 0.5 if returns had no memory (shuffled)")
    c[3].metric("H of |returns|", f"{h['h_vol']:.3f}", help="Memory in volatility; typically 0.7-0.9, which is why GARCH exists")
    if h["p"] < 0.05 and h["h_rs"] > 0.5:
        msg = "Returns are **persistent (trending)** beyond what chance would produce."
    elif h["p"] < 0.05:
        msg = "Returns are **anti-persistent (mean-reverting)** beyond what chance would produce."
    else:
        msg = "Returns are **statistically indistinguishable from a random walk** (H ~ 0.5)."
    st.markdown(msg + f" 95% band under no-memory: {h['null_lo']:.3f} - {h['null_hi']:.3f}. Raw (uncorrected) R/S slope: {h['h_rs_raw']:.3f}.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["ls"], y=h["lrs"], mode="markers+lines", name="log R/S (observed)", line=dict(color="#2a78d6")))
    fig.add_trace(go.Scatter(x=h["ls"], y=h["lers"], mode="lines", name="log E[R/S] (iid benchmark)", line=dict(color=MUTED, dash="dash")))
    style(fig, "R/S scaling: observed vs random-walk benchmark", height=320).update_xaxes(title="log window size")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Differencing order for stationarity")
    tbl, d_final, d_both, d_adf = stn
    st.dataframe(tbl.set_index("d").round(4), width="stretch")
    if d_both is not None:
        st.success(f"Both ADF (rejects unit root) and KPSS (fails to reject stationarity) agree at **d = {d_both}**.")
    elif d_adf is not None:
        st.warning(f"ADF says stationary at d = {d_adf}, but KPSS still rejects stationarity - ambiguous. Using d = {d_final}.")
    else:
        st.warning("No test pair agreed within d <= 2; defaulting to d = 1.")
    st.caption("Prices are almost always I(1): log price needs d = 1 and the log return is the stationary series. "
               "H and d together tell you whether anything is left to forecast.")


def section_arima(a):
    p, d, q = a["order"]
    st.markdown(f"Selected by BIC: **ARIMA({p},{d},{q})**, trend = `{a['trend']}`, on log price.")
    left, right = st.columns([1, 1])
    left.markdown("**Top specifications (BIC)**")
    left.dataframe(a["grid"].round(1), width="stretch", hide_index=True)
    pr = pd.DataFrame({"coef": a["params"], "p": a["pvalues"]}).round(4)
    right.markdown("**Coefficients**")
    right.dataframe(pr, width="stretch")

    h = len(a["fc_price"])
    xs = list(range(1, h + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=list(a["fc_hi"]) + list(a["fc_lo"][::-1]), fill="toself",
                             fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="95% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=a["fc_price"], name="ARIMA forecast", line=dict(color="#2a78d6", width=2)))
    fig.add_hline(y=a["last_price"], line_dash="dot", line_color=MUTED, annotation_text="last close")
    style(fig, f"{h}-day ARIMA forecast (price)", height=320).update_xaxes(title="trading days ahead")
    st.plotly_chart(fig, width="stretch")

    st.markdown(f"**Hold-out test** - last {a['k']} days, one-step-ahead, parameters frozen from the training window")
    c = st.columns(4)
    c[0].metric("RMSE ARIMA / random walk", f"{a['rmse_arima'] / a['rmse_zero']:.4f}", help="<1 means ARIMA beat a zero-return forecast")
    c[1].metric("RMSE ARIMA / drift-only", f"{a['rmse_arima'] / a['rmse_drift']:.4f}")
    c[2].metric("Directional hit rate", f"{a['hit_rate'] * 100:.1f}%", help=f"Binomial p vs 50% = {a['hit_p']:.3f}. Always-guess-up would score {a['up_rate'] * 100:.1f}%")
    c[3].metric("Residual Ljung-Box p (10)", f"{a['lb_p']:.3f}", help="High = no leftover linear autocorrelation")
    useful = a["rmse_arima"] / a["rmse_zero"] < 0.995 and a["hit_p"] < 0.05 and a["hit_rate"] > a["up_rate"]
    if useful:
        st.success("ARIMA shows out-of-sample skill over a random walk on this hold-out. Treat as weak and re-test on other windows.")
    else:
        st.info("**Not useful for point forecasts here**: no meaningful out-of-sample gain over a random walk / drift. "
                "Use it for the interval width and the diagnostics, not for direction.")
    st.caption(f"ARCH-LM p on residuals = {a['arch_p']:.3g}" + (" - strong volatility clustering left, which is what GARCH models." if a["arch_p"] < 0.05 else "."))


def section_vol(ret_pct, ewma_lam, garch_tbl, garch_name, gs, res_g):
    r = ret_pct.values
    cond, nxt = ewma_conditional_var(r, ewma_lam)
    ewma_ann = float(np.sqrt(nxt) / 100 * np.sqrt(TDAYS))
    lam_mle = ewma_lambda_mle(r)
    rv21 = float(ret_pct.tail(21).std() / 100 * np.sqrt(TDAYS))
    rv_all = float(ret_pct.std() / 100 * np.sqrt(TDAYS))
    c = st.columns(5)
    c[0].metric("EWMA vol (next day, ann.)", pct(ewma_ann), help=f"lambda = {ewma_lam:.3f}")
    c[1].metric("EWMA lambda (MLE)", f"{lam_mle:.3f}", help="Decay maximising the Gaussian likelihood on this window; RiskMetrics uses 0.94")
    c[2].metric(f"{garch_name} vol (next day)", pct(avg_vol(gs["exp_var"], 1)))
    c[3].metric("Realised 21d / full window", f"{pct(rv21, 0)} / {pct(rv_all, 0)}")
    c[4].metric("Persistence / half-life", f"{gs['persistence']:.3f} / {gs['half_life']:.0f}d" if np.isfinite(gs["half_life"]) else f"{gs['persistence']:.3f} / n.a.")

    st.markdown(f"**Model-implied volatility term structure** ({garch_name}, Student-t, nu = {gs['nu']:.1f}) - what the model expects the average annualised vol to be over each horizon")
    hs = [5, 21, 63, 126, 250]
    st.dataframe(pd.DataFrame({"Horizon (days)": hs, "GARCH-implied vol (ann.)": [pct(avg_vol(gs['exp_var'], h)) for h in hs],
                               "EWMA flat forecast (ann.)": [pct(ewma_ann)] * len(hs)}), hide_index=True, width="stretch")

    last = ret_pct.index[-1] - pd.DateOffset(years=2)
    mask = ret_pct.index > last
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ret_pct.index[mask], y=gs["cond_vol_ann"][mask], name=garch_name, line=dict(color="#2a78d6", width=2)))
    fig.add_trace(go.Scatter(x=ret_pct.index[mask], y=np.sqrt(cond[mask]) / 100 * np.sqrt(TDAYS), name=f"EWMA (λ={ewma_lam:.2f})", line=dict(color="#eb6834", width=2)))
    rv = ret_pct.rolling(21).std() / 100 * np.sqrt(TDAYS)
    fig.add_trace(go.Scatter(x=ret_pct.index[mask], y=rv[mask], name="Realised 21d", line=dict(color=MUTED, width=1)))
    style(fig, "Conditional volatility, last 2 years (annualised)", height=340).update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    left.markdown("**GARCH family ranked by BIC**")
    left.dataframe(garch_tbl.round(1), hide_index=True, width="stretch")
    right.markdown(f"**{garch_name} parameters**")
    right.dataframe(gs["params"].round(4), width="stretch")
    st.caption(f"Standardised-residual Ljung-Box (squares, 10 lags) p = {gs['lb_p']:.3f} - " + ("ARCH effects removed." if gs["lb_p"] > 0.05 else "some clustering remains; consider a longer window."))
    return ewma_ann


def section_mc(symbol, S0, ewma_ann, gs, res_g, garch_name, betas, factor_means_ann, hist_mean_ann):
    st.markdown(f"Last close **{S0:,.2f}**. Probability that the stock reaches your target within the horizon.")
    garch_vol_default = avg_vol(gs["exp_var"], 126)
    with st.form("mc_form"):
        c1, c2, c3 = st.columns(3)
        target = c1.number_input("Target price", min_value=0.01, value=round(S0 * 1.20, 2), step=1.0, key=f"tgt_{symbol}")
        H = c2.number_input("Horizon (trading days)", min_value=5, max_value=504, value=126, step=5, help="~21 trading days per month")
        N = c3.select_slider("Simulations", options=[5000, 10000, 20000, 50000], value=20000)
        v1, v2 = st.columns(2)
        vol_choice = v1.radio("Volatility source", [
            f"Model-implied (GARCH, path-dependent) - {pct(garch_vol_default)} avg over 126d",
            f"EWMA forecast - {pct(ewma_ann)}", "Manual input"])
        vol_manual = v1.number_input("Manual annualised vol (%)", min_value=1.0, max_value=300.0, value=float(round(garch_vol_default * 100, 1)), step=1.0)
        rf = v2.number_input("Risk-free rate for drift options (% p.a.)", value=6.5, step=0.25)
        f_drift = rf / 100 + float((betas * factor_means_ann).sum()) / 100 if betas is not None else None
        dopts = ["Zero drift (pure volatility view)", f"Risk-free rate - {rf:.1f}%",
                 f"Historical mean of this window - {hist_mean_ann * 100:.1f}%"]
        if f_drift is not None:
            dopts.append(f"Factor-model expected return (rf + beta x long-run factor premia) - {f_drift * 100:.1f}%")
        dopts.append("Manual input")
        drift_choice = v2.radio("Drift (expected return)", dopts)
        drift_manual = v2.number_input("Manual drift (% p.a.)", value=12.0, step=1.0)
        t1, t2 = st.columns(2)
        fat = t1.checkbox("Fat-tailed shocks (Student-t) for constant-vol modes", value=True)
        nu = t2.number_input("Student-t degrees of freedom", min_value=3.0, max_value=60.0, value=float(round(min(max(gs["nu"], 3.0), 60.0), 1)), step=0.5)
        seed = st.number_input("Random seed", value=42, step=1)
        go_btn = st.form_submit_button("Run Monte Carlo", type="primary")
    if not go_btn and "mc_last" not in st.session_state:
        st.info("Set the inputs and press **Run Monte Carlo**.")
        return
    if go_btn:
        if vol_choice.startswith("Model"):
            mode, sig, vlabel = "garch", None, f"{garch_name} path-dependent vol"
        elif vol_choice.startswith("EWMA"):
            mode, sig, vlabel = "const", ewma_ann, f"EWMA vol {pct(ewma_ann)}"
        else:
            mode, sig, vlabel = "const", vol_manual / 100, f"manual vol {vol_manual:.1f}%"
        if drift_choice.startswith("Zero"):
            mu = 0.0
        elif drift_choice.startswith("Risk"):
            mu = rf / 100
        elif drift_choice.startswith("Historical"):
            mu = hist_mean_ann
        elif drift_choice.startswith("Factor"):
            mu = f_drift
        else:
            mu = drift_manual / 100
        with st.spinner("Simulating..."):
            out = mc_run(S0, target, int(H), int(N), mu, mode, sig_ann=sig, nu_const=(float(nu) if fat else None),
                         garch_res=res_g, seed=int(seed))
        st.session_state["mc_last"] = dict(out=out, target=target, H=int(H), mu=mu, vlabel=vlabel, mode=mode, sig=sig,
                                           fat=fat, S0=S0, N=int(N), symbol=symbol)
    m = st.session_state["mc_last"]
    if m["symbol"] != symbol:
        st.info("Inputs changed for a new ticker - press **Run Monte Carlo** again.")
        return
    o, T, H, mu = m["out"], m["target"], m["H"], m["mu"]
    word = "at or above" if o["up"] else "at or below"
    c = st.columns(4)
    c[0].metric(f"P(closes {word} {T:,.2f} on day {H})", pct(o["p_term"]), help=f"MC standard error ±{o['se_term'] * 100:.2f} pts")
    c[1].metric("P(touches target on any close)", pct(o["p_touch"]), help=f"MC standard error ±{o['se_touch'] * 100:.2f} pts")
    c[2].metric("Median day first reached", "-" if np.isnan(o["median_hit_day"]) else f"{o['median_hit_day']:.0f}")
    c[3].metric("Expected price at horizon", f"{o['mean_term']:,.2f}", f"{(o['mean_term'] / m['S0'] - 1) * 100:+.1f}%")
    st.caption(f"Assumptions: {m['vlabel']}; drift {mu * 100:.1f}% p.a.; {m['N']:,} paths; horizon {H} trading days (~{H / 21:.1f} months)."
               + ("" if m["mode"] == "garch" else (" Student-t shocks." if m["fat"] else " Normal shocks.")))
    if m["mode"] == "const" and not m["fat"]:
        st.caption(f"Analytic lognormal cross-check for the terminal probability: {pct(lognormal_term_prob(m['S0'], T, H, mu, m['sig']))}.")

    xs = list(range(1, H + 1))
    f = o["fan"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=list(f[95]) + list(f[5][::-1]), fill="toself", fillcolor="rgba(42,120,214,0.12)", line=dict(width=0), name="5-95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=list(f[75]) + list(f[25][::-1]), fill="toself", fillcolor="rgba(42,120,214,0.25)", line=dict(width=0), name="25-75%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=f[50], name="Median", line=dict(color="#2a78d6", width=2)))
    fig.add_hline(y=T, line_color="#eb6834", line_dash="dash", annotation_text="target")
    fig.add_hline(y=m["S0"], line_color=MUTED, line_dash="dot", annotation_text="last close")
    style(fig, "Simulated price fan", height=360).update_xaxes(title="trading days ahead")
    left, right = st.columns(2)
    left.plotly_chart(fig, width="stretch")

    fig = go.Figure(go.Scatter(x=xs, y=o["cum_hit"], line=dict(color="#eb6834", width=2), fill="tozeroy", fillcolor="rgba(235,104,52,0.12)", name="P(hit by day)"))
    style(fig, "Cumulative probability of touching the target", height=360, legend=False).update_yaxes(tickformat=".0%", range=[0, max(0.05, o["cum_hit"].max() * 1.15)]).update_xaxes(title="trading days ahead")
    right.plotly_chart(fig, width="stretch")

    fig = go.Figure(go.Histogram(x=o["term"], nbinsx=80, marker_color="#2a78d6", opacity=0.85, name="terminal price"))
    fig.add_vline(x=T, line_color="#eb6834", line_dash="dash", annotation_text="target")
    style(fig, "Distribution of price at the horizon", height=300, legend=False).update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    q = o["quantiles"]
    st.dataframe(pd.DataFrame({"Percentile": ["5th", "25th", "50th", "75th", "95th"], "Price": [round(q[k], 2) for k in (5, 25, 50, 75, 95)],
                               "vs last close": [f"{(q[k] / m['S0'] - 1) * 100:+.1f}%" for k in (5, 25, 50, 75, 95)]}), hide_index=True, width="stretch")

    # sensitivity: analytic lognormal terminal probability across vol and drift
    ref_vol = m["sig"] if m["sig"] is not None else avg_vol(gs["exp_var"], H)
    vm = [0.6, 0.8, 1.0, 1.25, 1.5]
    dg = [0.0, 0.05, 0.10, 0.15, 0.20]
    z = [[lognormal_term_prob(m["S0"], T, H, d_, ref_vol * v_) for d_ in dg] for v_ in vm]
    fig = go.Figure(go.Heatmap(z=z, x=[f"{d_ * 100:.0f}%" for d_ in dg], y=[f"{ref_vol * v_ * 100:.0f}%" for v_ in vm],
                               text=[[f"{x * 100:.0f}%" for x in row] for row in z], texttemplate="%{text}", colorscale="Blues", showscale=False))
    style(fig, "Sensitivity of P(terminal) to your two big assumptions (lognormal approx.)", height=300, legend=False).update_layout(hovermode="closest")
    fig.update_xaxes(title="annual drift").update_yaxes(title="annual vol")
    st.plotly_chart(fig, width="stretch")
    st.caption("Probabilities are conditional on the chosen drift and volatility, and ignore jumps (results, gaps), liquidity and path-dependent events. "
               "Drift is the dominant unknown over months-long horizons, so read the heat-map, not just the single number.")


# ============================================================================= APP
def main():
    st.set_page_config(page_title="Factor Lab", page_icon="📈", layout="wide")
    st.title("Factor Lab")
    st.caption("Four-factor decomposition (IIM-A data) · memory & stationarity · ARIMA · EWMA/GARCH · Monte Carlo target odds")

    # ---- factor data: auto-load, uploader is an optional override
    path = find_factor_file()
    with st.sidebar:
        up = st.file_uploader("Factor CSV (optional override)", type="csv", help="Only needed to use a newer IIM-A release than the file saved next to app.py")
    if up is not None:
        fac = read_factor_bytes(up.getvalue())
        src = f"uploaded: {up.name}"
    elif path is not None:
        fac = read_factor_path(str(path), path.stat().st_mtime)
        src = path.name
    else:
        st.error("No factor file found. Save the IIM-A CSV next to app.py (or in ./data/) or upload it in the sidebar.")
        st.stop()
    st.sidebar.success(f"Factors: {fac.index.min():%d %b %Y} - {fac.index.max():%d %b %Y}\n\n{src}")

    with st.sidebar.form("inputs"):
        raw = st.text_input("NSE / BSE ticker", value="RELIANCE", help="e.g. TCS, HDFCBANK, M&M, or a BSE code like 500325")
        exch = st.radio("Exchange", ["NSE", "BSE"], horizontal=True)
        se_key = st.selectbox("Standard errors", list(SE_TYPES), index=0)
        lb_reg = st.selectbox("Regression window (ends at last factor date)", list(LOOKBACKS), index=2)
        lb_ts = st.selectbox("Time-series window (ends today)", list(LOOKBACKS), index=2)
        lam = st.slider("EWMA lambda", 0.85, 0.99, 0.94, 0.01)
        st.form_submit_button("Run analysis", type="primary")

    symbol = resolve_symbol(raw, exch)
    try:
        px = load_prices(symbol)
    except Exception as e:
        st.error(f"Could not load **{symbol}**: {e}. Check the ticker / exchange suffix.")
        st.stop()
    st.markdown(f"### {symbol} · last close {px.iloc[-1]:,.2f} on {px.index[-1]:%d %b %Y} · {len(px):,} daily bars since {px.index[0]:%b %Y}")

    simple = px.pct_change().dropna() * 100  # % simple returns (regression)
    pxw = window(px, LOOKBACKS[lb_ts])
    logp = np.log(pxw)
    logret = logp.diff().dropna() * 100  # % log returns (time-series work)
    if len(logret) < 250:
        st.error("Fewer than 250 daily returns in the time-series window - pick a longer window or a longer-listed stock.")
        st.stop()

    tabs = st.tabs(["1 · Factors", "2 · Memory & stationarity", "3 · ARIMA", "4 · Volatility", "5 · Monte Carlo"])
    reg = None
    with tabs[0]:
        try:
            reg = factor_regression(simple, fac, se_key, LOOKBACKS[lb_reg])
            section_factors(reg, symbol)
        except Exception as e:
            st.error(f"Factor regression failed: {e}")
        else:
            if reg["diag"]["end"] < px.index[-1] - pd.Timedelta(days=45):
                st.info(f"The factor file ends {reg['diag']['end']:%d %b %Y}; the regression uses data up to then. Tabs 2-5 use prices up to {px.index[-1]:%d %b %Y}.")
    with tabs[1]:
        try:
            h = hurst_analysis(logret)
            stn = stationarity_analysis(logp)
            section_memory(h, stn)
        except Exception as e:
            stn = (None, 1, 1, 1)
            st.error(f"Memory / stationarity failed: {e}")
    with tabs[2]:
        try:
            section_arima(arima_analysis(logp, int(stn[1])))
        except Exception as e:
            st.error(f"ARIMA failed: {e}")
    gs = res_g = ewma_ann = None
    with tabs[3]:
        try:
            g_tbl, g_name = garch_selection(logret)
            res_g = fit_garch(logret, g_name)
            gs = garch_summary(res_g, g_name)
            ewma_ann = section_vol(logret, lam, g_tbl, g_name, gs, res_g)
        except Exception as e:
            st.error(f"Volatility models failed: {e}")
    with tabs[4]:
        if gs is None:
            st.warning("Monte Carlo needs the volatility models (tab 4) to have run.")
        else:
            try:
                hist_mean = float(np.expm1(logret.mean() / 100 * TDAYS + 0.5 * (logret.std() / 100) ** 2 * TDAYS))
                section_mc(symbol, float(px.iloc[-1]), ewma_ann, gs, res_g, g_name,
                           reg["betas"] if reg else None, reg["factor_means_ann"] if reg else None, hist_mean)
            except Exception as e:
                st.error(f"Monte Carlo failed: {e}")


if __name__ == "__main__":
    main()
