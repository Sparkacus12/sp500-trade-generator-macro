import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.stats import shapiro, linregress
@st.cache_data(ttl=60 * 60 * 12)
def get_fred_data():
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=5)

    series = {}

    for name, code in FRED_SERIES.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
            df = pd.read_csv(url)
            df.columns = ["Date", name]
            df["Date"] = pd.to_datetime(df["Date"])
            df[name] = pd.to_numeric(df[name], errors="coerce")
            df = df[(df["Date"] >= start) & (df["Date"] <= end)]
            series[name] = df.set_index("Date")[name]
        except Exception:
            series[name] = pd.Series(dtype=float)

    fred = pd.DataFrame(series).sort_index().ffill()
    return fred

# ============================================================
# Settings
# ============================================================
st.set_page_config(page_title="S&P 500 Trade Generator", layout="wide")

LOOKBACK = 30
P_THRESHOLD = 0.10
PRICE_PERIOD = "3y"
MAX_LONGS = 3
MAX_SHORTS = 3

st.title("S&P 500 Trade Generator")
st.caption(
    "Systematic screen only, not investment advice. Uses Yahoo Finance prices plus FRED macro data. "
    "Normality test is Shapiro-Wilk on rolling 30-day returns."
)

# ============================================================
# Data
# ============================================================
@st.cache_data(ttl=60 * 60 * 12)
def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=20).text
    table = pd.read_html(StringIO(html))[0]
    table["Ticker"] = table["Symbol"].str.replace(".", "-", regex=False)
    return table[["Ticker", "Security", "GICS Sector"]]

@st.cache_data(ttl=60 * 60 * 6)
def get_prices(tickers):
    data = yf.download(tickers, period=PRICE_PERIOD, auto_adjust=True, progress=False, threads=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    return close.dropna(axis=1, how="all").ffill()

MARKET_TICKERS = {
    "Market": "SPY", "Credit": "HYG", "Rates": "TLT", "Dollar": "UUP", "Oil ETF": "USO",
    "Communication Services": "XLC", "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
    "Energy": "XLE", "Financials": "XLF", "Health Care": "XLV", "Industrials": "XLI",
    "Information Technology": "XLK", "Materials": "XLB", "Real Estate": "XLRE", "Utilities": "XLU",
}

@st.cache_data(ttl=60 * 60 * 6)
def get_market_proxy_prices():
    data = yf.download(list(MARKET_TICKERS.values()), period=PRICE_PERIOD, auto_adjust=True, progress=False, threads=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    return close.dropna(axis=1, how="all").ffill()

FRED_SERIES = {
    "industrial_production": "INDPRO",
    "retail_sales": "RSAFS",
    "payrolls": "PAYEMS",
    "unemployment": "UNRATE",
    "cpi": "CPIAUCSL",
    "ppi": "PPIACO",
    "ten_year": "DGS10",
    "two_year": "DGS2",
    "hy_spread": "BAMLH0A0HYM2",
    "financial_conditions": "NFCI",
    "m2": "M2SL",
    "dollar": "DTWEXBGS",
    "oil": "DCOILWTICO",
}

@st.cache_data(ttl=60 * 60 * 12)
def get_fred_data():
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=5)
    series = {}
    for name, code in FRED_SERIES.items():
        try:
            s = pdr.DataReader(code, "fred", start, end)
            series[name] = s.iloc[:, 0]
        except Exception:
            series[name] = pd.Series(dtype=float)
    return pd.DataFrame(series).sort_index().ffill()

# ============================================================
# Helpers
# ============================================================
def trend_score(series):
    series = series.dropna()
    if len(series) < 3:
        return np.nan
    y = np.log(series.values)
    x = np.arange(len(y))
    return linregress(x, y).slope * 100

def normality_pvalue(series):
    returns = series.pct_change().dropna()
    if len(returns) < LOOKBACK - 1:
        return np.nan
    try:
        return shapiro(returns).pvalue
    except Exception:
        return np.nan

def pct_return(price_df, ticker, days=30):
    try:
        if ticker is None or ticker not in price_df.columns:
            return 0.0
        s = price_df[ticker].dropna()
        if len(s) < days + 1:
            return 0.0
        return s.iloc[-1] / s.iloc[-days] - 1
    except Exception:
        return 0.0

def realised_vol(series):
    r = series.pct_change().dropna()
    return r.std() if len(r) >= 3 else 0.0

def vol_compression_for_ticker(ticker, prices):
    try:
        s = prices[ticker].dropna()
        if len(s) < LOOKBACK:
            return 0.0
        return realised_vol(s.iloc[-30:-15]) - realised_vol(s.iloc[-15:])
    except Exception:
        return 0.0

def zscore_latest(series, lookback=36, transform="diff"):
    s = series.dropna()
    if len(s) < 15:
        return 0.0
    if transform == "yoy":
        x = s.pct_change(12).dropna()
    elif transform == "diff":
        x = s.diff(3).dropna()
    elif transform == "level":
        x = s.copy()
    elif transform == "inverse_level":
        x = -s.copy()
    else:
        x = s.copy()
    x = x.dropna()
    if len(x) < 12:
        return 0.0
    window = x.iloc[-lookback:] if len(x) >= lookback else x
    std = window.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((window.iloc[-1] - window.mean()) / std)

def build_fred_macro_factor_scores(fred):
    if fred.empty:
        return {k: 0.0 for k in ["Growth", "Inflation relief", "Financial conditions", "Liquidity", "Dollar relief", "Oil", "Yield curve"]}
    growth = np.nanmean([
        zscore_latest(fred.get("industrial_production", pd.Series(dtype=float)), transform="yoy"),
        zscore_latest(fred.get("retail_sales", pd.Series(dtype=float)), transform="yoy"),
        zscore_latest(fred.get("payrolls", pd.Series(dtype=float)), transform="diff"),
        -zscore_latest(fred.get("unemployment", pd.Series(dtype=float)), transform="diff"),
    ])
    inflation_relief = -np.nanmean([
        zscore_latest(fred.get("cpi", pd.Series(dtype=float)), transform="diff"),
        zscore_latest(fred.get("ppi", pd.Series(dtype=float)), transform="diff"),
    ])
    financial_conditions = np.nanmean([
        -zscore_latest(fred.get("hy_spread", pd.Series(dtype=float)), transform="level"),
        -zscore_latest(fred.get("financial_conditions", pd.Series(dtype=float)), transform="level"),
    ])
    liquidity = zscore_latest(fred.get("m2", pd.Series(dtype=float)), transform="yoy")
    dollar_relief = -zscore_latest(fred.get("dollar", pd.Series(dtype=float)), transform="diff")
    oil = zscore_latest(fred.get("oil", pd.Series(dtype=float)), transform="diff")
    try:
        curve = (fred["ten_year"] - fred["two_year"]).dropna()
        yield_curve = zscore_latest(curve, transform="level")
    except Exception:
        yield_curve = 0.0
    return {
        "Growth": float(np.nan_to_num(growth)),
        "Inflation relief": float(np.nan_to_num(inflation_relief)),
        "Financial conditions": float(np.nan_to_num(financial_conditions)),
        "Liquidity": float(np.nan_to_num(liquidity)),
        "Dollar relief": float(np.nan_to_num(dollar_relief)),
        "Oil": float(np.nan_to_num(oil)),
        "Yield curve": float(np.nan_to_num(yield_curve)),
    }

SECTOR_FRED_WEIGHTS = {
    "Information Technology": {"Growth": .25, "Inflation relief": .20, "Financial conditions": .20, "Liquidity": .25, "Dollar relief": .10},
    "Communication Services": {"Growth": .30, "Inflation relief": .15, "Financial conditions": .15, "Liquidity": .20, "Dollar relief": .20},
    "Consumer Discretionary": {"Growth": .35, "Inflation relief": .20, "Financial conditions": .20, "Liquidity": .15, "Dollar relief": .10},
    "Consumer Staples": {"Growth": .10, "Inflation relief": .35, "Financial conditions": .15, "Liquidity": .10, "Dollar relief": .30},
    "Industrials": {"Growth": .45, "Inflation relief": .10, "Financial conditions": .20, "Liquidity": .10, "Dollar relief": .15},
    "Materials": {"Growth": .45, "Inflation relief": .05, "Financial conditions": .15, "Liquidity": .10, "Dollar relief": .15, "Oil": .10},
    "Energy": {"Growth": .15, "Inflation relief": -.10, "Financial conditions": .10, "Liquidity": .05, "Dollar relief": .10, "Oil": .70},
    "Financials": {"Growth": .25, "Inflation relief": .05, "Financial conditions": .25, "Liquidity": .05, "Yield curve": .40},
    "Health Care": {"Growth": .10, "Inflation relief": .25, "Financial conditions": .15, "Liquidity": .15, "Dollar relief": .35},
    "Real Estate": {"Growth": .10, "Inflation relief": .30, "Financial conditions": .25, "Liquidity": .25, "Yield curve": -.10},
    "Utilities": {"Growth": -.05, "Inflation relief": .35, "Financial conditions": .25, "Liquidity": .25, "Yield curve": -.10},
}

def fred_sector_macro_score(sector, fred_scores):
    weights = SECTOR_FRED_WEIGHTS.get(sector, {"Growth": .30, "Inflation relief": .20, "Financial conditions": .20, "Liquidity": .15, "Dollar relief": .15})
    return sum(weights.get(k, 0.0) * fred_scores.get(k, 0.0) for k in weights)

def market_macro_score(sector, ticker, prices, market_prices):
    sector_etf = MARKET_TICKERS.get(sector)
    sector_mom = pct_return(market_prices, sector_etf)
    market_mom = pct_return(market_prices, "SPY")
    credit_mom = pct_return(market_prices, "HYG")
    rates_mom = pct_return(market_prices, "TLT")
    dollar_mom = pct_return(market_prices, "UUP")
    oil_mom = pct_return(market_prices, "USO")
    score = 45 * sector_mom + 25 * market_mom + 20 * credit_mom - 10 * rates_mom - 10 * dollar_mom
    if sector == "Energy":
        score += 25 * oil_mom
    stock_ret = pct_return(prices, ticker)
    sector_ret = pct_return(market_prices, sector_etf)
    return score + (stock_ret - sector_ret) * 100

def combined_macro_earnings_score(sector, ticker, prices, market_prices, fred_scores):
    fred_score = fred_sector_macro_score(sector, fred_scores)
    mkt_score = market_macro_score(sector, ticker, prices, market_prices)
    return 2.0 * fred_score + 0.35 * mkt_score, fred_score, mkt_score

def score_to_signal(score):
    if score > 2:
        return "Positive macro-earnings backdrop"
    if score < -2:
        return "Negative macro-earnings backdrop"
    return "Neutral"

def implied_earnings_revision_score(row):
    return 40 * row["Relative strength vs sector"] + 25 * row["Relative strength vs market"] + 20 * row["Trend score"] + 15 * row["Vol compression"] + 10 * row["Combined macro earnings score"]

def performance_stats(bt):
    bt = bt.dropna().copy()
    if bt.empty:
        return bt, np.nan, np.nan, np.nan, np.nan, np.nan
    bt["Equity curve"] = (1 + bt["Return"]).cumprod()
    total_return = bt["Equity curve"].iloc[-1] - 1
    ann_return = bt["Equity curve"].iloc[-1] ** (252 / len(bt)) - 1
    ann_vol = bt["Return"].std() * np.sqrt(252)
    sharpe = bt["Return"].mean() / bt["Return"].std() * np.sqrt(252) if bt["Return"].std() != 0 else np.nan
    dd = bt["Equity curve"] / bt["Equity curve"].cummax() - 1
    return bt, total_return, ann_return, ann_vol, sharpe, dd.min()

def show_backtest(name, bt):
    st.markdown(f"### {name}")
    if bt.empty:
        st.write("No results generated.")
        return
    bt, total_return, ann_return, ann_vol, sharpe, max_dd = performance_stats(bt)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total return", f"{total_return:.2%}")
    c2.metric("Annualised return", f"{ann_return:.2%}")
    c3.metric("Annualised vol", f"{ann_vol:.2%}")
    c4.metric("Sharpe ratio", f"{sharpe:.2f}")
    c5.metric("Max drawdown", f"{max_dd:.2%}")
    st.line_chart(bt.set_index("Date")["Equity curve"])
    st.dataframe(bt.tail(20), use_container_width=True)

# ============================================================
# Load
# ============================================================
with st.spinner("Loading S&P 500, prices, market proxies and FRED macro data..."):
    sp500 = get_sp500()
    tickers = sp500["Ticker"].tolist()
    prices = get_prices(tickers)
    market_prices = get_market_proxy_prices()
    fred = get_fred_data()

fred_scores = build_fred_macro_factor_scores(fred)
available = [t for t in tickers if t in prices.columns]

st.subheader("FRED macro regime dashboard")
st.dataframe(pd.DataFrame([{"Factor": k, "Latest z-score": v} for k, v in fred_scores.items()]), use_container_width=True)
st.write("FRED macro factors are mapped to sector earnings sensitivity weights, then combined with market-implied confirmation.")

# ============================================================
# Current screen
# ============================================================
rows = []
for ticker in available:
    s = prices[ticker].dropna()
    if len(s) < LOOKBACK + 1:
        continue
    window = s.iloc[-LOOKBACK:]
    pval = normality_pvalue(window)
    trend = trend_score(window)
    if np.isnan(pval) or np.isnan(trend):
        continue
    rows.append({
        "Ticker": ticker,
        "Trend score": trend,
        "30d return": window.iloc[-1] / window.iloc[0] - 1,
        "Normality p-value": pval,
        "Pass normality": pval > P_THRESHOLD,
        "Vol compression": vol_compression_for_ticker(ticker, prices),
    })

df = pd.DataFrame(rows).merge(sp500, on="Ticker", how="left")
macro_parts = df.apply(lambda r: combined_macro_earnings_score(r["GICS Sector"], r["Ticker"], prices, market_prices, fred_scores), axis=1)
df["Combined macro earnings score"] = [x[0] for x in macro_parts]
df["FRED sector macro score"] = [x[1] for x in macro_parts]
df["Market-implied macro score"] = [x[2] for x in macro_parts]
df["Macro signal"] = df["Combined macro earnings score"].apply(score_to_signal)
market_30d_return = pct_return(market_prices, "SPY")
df["Relative strength vs market"] = df["30d return"] - market_30d_return
df["Relative strength vs sector"] = df.apply(lambda r: r["30d return"] - pct_return(market_prices, MARKET_TICKERS.get(r["GICS Sector"])), axis=1)
df["Implied earnings revision score"] = df.apply(implied_earnings_revision_score, axis=1)
df["Implied earnings revision signal"] = np.where(df["Implied earnings revision score"] > 2, "Likely upward revision pressure", np.where(df["Implied earnings revision score"] < -2, "Likely downward revision pressure", "Neutral"))

passed = df[df["Pass normality"]]
buys = passed[passed["Trend score"] > 0].sort_values("Implied earnings revision score", ascending=False).head(MAX_LONGS)
sells = passed[passed["Trend score"] < 0].sort_values("Implied earnings revision score", ascending=True).head(MAX_SHORTS)

display_cols = ["Ticker", "Security", "GICS Sector", "Trend score", "30d return", "Normality p-value", "FRED sector macro score", "Market-implied macro score", "Combined macro earnings score", "Relative strength vs sector", "Vol compression", "Implied earnings revision score", "Implied earnings revision signal"]

st.subheader("Strategy 1: Core daily trend screen â no backtest")
c1, c2 = st.columns(2)
with c1:
    st.markdown("### Buys")
    st.dataframe(buys[display_cols], use_container_width=True)
with c2:
    st.markdown("### Sells")
    st.dataframe(sells[display_cols], use_container_width=True)

st.subheader("Implied earnings revision overlay")
revision_table = df[["Ticker", "Security", "GICS Sector", "30d return", "Relative strength vs market", "Relative strength vs sector", "Trend score", "Vol compression", "FRED sector macro score", "Market-implied macro score", "Combined macro earnings score", "Implied earnings revision score", "Implied earnings revision signal"]].sort_values("Implied earnings revision score", ascending=False)
c1, c2 = st.columns(2)
with c1:
    st.markdown("### Highest implied upward revision pressure")
    st.dataframe(revision_table.head(15), use_container_width=True)
with c2:
    st.markdown("### Highest implied downward revision pressure")
    st.dataframe(revision_table.tail(15).sort_values("Implied earnings revision score"), use_container_width=True)

# ============================================================
# Strategy 2: trend-break watchlist
# ============================================================
st.subheader("Strategy 2: Trend-break watchlist")
down_breaks, up_breaks = [], []
for ticker in available:
    s = prices[ticker].dropna()
    if len(s) < LOOKBACK + 2:
        continue
    prior = s.iloc[-LOOKBACK - 1:-1]
    current = s.iloc[-LOOKBACK:]
    prior_p = normality_pvalue(prior)
    current_p = normality_pvalue(current)
    prior_trend = trend_score(prior)
    last_move = s.iloc[-1] / s.iloc[-2] - 1
    if np.isnan(prior_p) or np.isnan(current_p) or np.isnan(prior_trend):
        continue
    if prior_trend > 0 and prior_p > P_THRESHOLD and current_p <= P_THRESHOLD and last_move < 0:
        down_breaks.append({"Ticker": ticker, "t-1 trend score": prior_trend, "Last-day move": last_move, "t-1 p-value": prior_p, "Current p-value": current_p})
    if prior_trend < 0 and prior_p > P_THRESHOLD and current_p <= P_THRESHOLD and last_move > 0:
        up_breaks.append({"Ticker": ticker, "t-1 trend score": prior_trend, "Last-day move": last_move, "t-1 p-value": prior_p, "Current p-value": current_p})

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Possible sells")
    st.dataframe(pd.DataFrame(down_breaks).merge(sp500, on="Ticker", how="left") if down_breaks else pd.DataFrame(), use_container_width=True)
with c2:
    st.markdown("### Possible buys")
    st.dataframe(pd.DataFrame(up_breaks).merge(sp500, on="Ticker", how="left") if up_breaks else pd.DataFrame(), use_container_width=True)

# ============================================================
# Backtests: strategy 2 and 3 only
# ============================================================
st.subheader("Backtests")
if not st.button("Run Strategy 2 and Strategy 3 backtests"):
    st.info("Click to run the backtests. First run may take a few minutes.")
    st.stop()

@st.cache_data(ttl=60 * 60 * 6, show_spinner=True)
def run_trend_break_backtest(prices, lookback=30, p_threshold=0.10):
    returns = prices.pct_change()
    results = []
    for i in range(lookback + 2, len(prices) - 1):
        trade_date = prices.index[i + 1]
        longs, shorts = [], []
        for ticker in prices.columns:
            prior = prices[ticker].iloc[i - lookback - 1:i - 1].dropna()
            current = prices[ticker].iloc[i - lookback:i].dropna()
            if len(prior) < lookback or len(current) < lookback:
                continue
            prior_p = normality_pvalue(prior)
            current_p = normality_pvalue(current)
            prior_trend = trend_score(prior)
            last_move = prices[ticker].iloc[i - 1] / prices[ticker].iloc[i - 2] - 1
            if np.isnan(prior_p) or np.isnan(current_p) or np.isnan(prior_trend):
                continue
            if prior_trend > 0 and prior_p > p_threshold and current_p <= p_threshold and last_move < 0:
                shorts.append(ticker)
            if prior_trend < 0 and prior_p > p_threshold and current_p <= p_threshold and last_move > 0:
                longs.append(ticker)
        next_returns = returns.loc[trade_date]
        results.append({"Date": trade_date, "Return": (next_returns[longs].mean() if longs else 0) - (next_returns[shorts].mean() if shorts else 0), "Number longs": len(longs), "Number shorts": len(shorts), "Longs": ", ".join(longs), "Shorts": ", ".join(shorts)})
    return pd.DataFrame(results)

@st.cache_data(ttl=60 * 60 * 6, show_spinner=True)
def run_normalisation_backtest(prices, sector_table, market_prices, fred_scores, lookback=30, p_threshold=0.10):
    returns = prices.pct_change()
    positions, results, trade_log = {}, [], []
    sector_lookup = sector_table.set_index("Ticker")["GICS Sector"].to_dict()
    for i in range(lookback + 2, len(prices) - 1):
        signal_date, trade_date = prices.index[i], prices.index[i + 1]
        for ticker in list(positions.keys()):
            current = prices[ticker].iloc[i - lookback:i].dropna()
            if len(current) < lookback:
                del positions[ticker]
                continue
            pval, trend = normality_pvalue(current), trend_score(current)
            ret_30d = current.iloc[-1] / current.iloc[0] - 1
            sector = sector_lookup.get(ticker, None)
            macro_total, fred_score, market_score = combined_macro_earnings_score(sector, ticker, prices.iloc[:i], market_prices.iloc[:i], fred_scores)
            rs_market = ret_30d - pct_return(market_prices.iloc[:i], "SPY")
            rs_sector = ret_30d - pct_return(market_prices.iloc[:i], MARKET_TICKERS.get(sector))
            implied_score = 40 * rs_sector + 25 * rs_market + 20 * trend + 15 * (realised_vol(current.iloc[:15]) - realised_vol(current.iloc[-15:])) + 10 * macro_total
            side = positions[ticker]
            exit_reason = None
            if pval <= p_threshold:
                exit_reason = "Normality broken"
            elif side == "LONG" and trend <= 0:
                exit_reason = "Positive trend broken"
            elif side == "SHORT" and trend >= 0:
                exit_reason = "Negative trend broken"
            elif side == "LONG" and implied_score <= 0:
                exit_reason = "Implied revision score no longer supports long"
            elif side == "SHORT" and implied_score >= 0:
                exit_reason = "Implied revision score no longer supports short"
            if exit_reason:
                trade_log.append({"Date": signal_date, "Ticker": ticker, "Action": "EXIT", "Side": side, "Reason": exit_reason, "P-value": pval, "Trend": trend, "Implied revision score": implied_score, "FRED macro score": fred_score, "Market macro score": market_score, "Combined macro score": macro_total})
                del positions[ticker]
        new_longs, new_shorts = [], []
        for ticker in prices.columns:
            if ticker in positions:
                continue
            prior = prices[ticker].iloc[i - lookback - 1:i - 1].dropna()
            current = prices[ticker].iloc[i - lookback:i].dropna()
            if len(prior) < lookback or len(current) < lookback:
                continue
            prior_p, current_p = normality_pvalue(prior), normality_pvalue(current)
            if np.isnan(prior_p) or np.isnan(current_p) or not (prior_p <= p_threshold and current_p > p_threshold):
                continue
            trend = trend_score(current)
            ret_30d = current.iloc[-1] / current.iloc[0] - 1
            sector = sector_lookup.get(ticker, None)
            macro_total, fred_score, market_score = combined_macro_earnings_score(sector, ticker, prices.iloc[:i], market_prices.iloc[:i], fred_scores)
            rs_market = ret_30d - pct_return(market_prices.iloc[:i], "SPY")
            rs_sector = ret_30d - pct_return(market_prices.iloc[:i], MARKET_TICKERS.get(sector))
            implied_score = 40 * rs_sector + 25 * rs_market + 20 * trend + 15 * (realised_vol(current.iloc[:15]) - realised_vol(current.iloc[-15:])) + 10 * macro_total
            if trend > 0 and ret_30d > 0 and macro_total > 0 and implied_score > 0:
                new_longs.append({"Ticker": ticker, "Rank score": implied_score, "P-value": current_p, "Trend": trend, "Implied revision score": implied_score, "FRED macro score": fred_score, "Market macro score": market_score, "Combined macro score": macro_total})
            elif trend < 0 and ret_30d < 0 and macro_total < 0 and implied_score < 0:
                new_shorts.append({"Ticker": ticker, "Rank score": abs(implied_score), "P-value": current_p, "Trend": trend, "Implied revision score": implied_score, "FRED macro score": fred_score, "Market macro score": market_score, "Combined macro score": macro_total})
        for side_name, candidates, side in [("LONG", new_longs, "LONG"), ("SHORT", new_shorts, "SHORT")]:
            if candidates:
                cand = pd.DataFrame(candidates).sort_values("Rank score", ascending=False).head(MAX_LONGS if side == "LONG" else MAX_SHORTS)
                for _, row in cand.iterrows():
                    positions[row["Ticker"]] = side
                    trade_log.append({"Date": signal_date, "Ticker": row["Ticker"], "Action": "ENTER", "Side": side, "Reason": "Normalised with revision pressure", "P-value": row["P-value"], "Trend": row["Trend"], "Implied revision score": row["Implied revision score"], "FRED macro score": row["FRED macro score"], "Market macro score": row["Market macro score"], "Combined macro score": row["Combined macro score"]})
        next_returns = returns.loc[trade_date]
        long_tickers = [t for t, side in positions.items() if side == "LONG"]
        short_tickers = [t for t, side in positions.items() if side == "SHORT"]
        results.append({"Date": trade_date, "Return": (next_returns[long_tickers].mean() if long_tickers else 0) - (next_returns[short_tickers].mean() if short_tickers else 0), "Number longs": len(long_tickers), "Number shorts": len(short_tickers), "Longs": ", ".join(long_tickers), "Shorts": ", ".join(short_tickers)})
    return pd.DataFrame(results), pd.DataFrame(trade_log)

bt2 = run_trend_break_backtest(prices, LOOKBACK, P_THRESHOLD)
show_backtest("Strategy 2: One-day trend-break reversal backtest", bt2)

bt3, trades3 = run_normalisation_backtest(prices, sp500, market_prices, fred_scores, LOOKBACK, P_THRESHOLD)
show_backtest("Strategy 3: Normalisation regime shift, hold until break", bt3)

st.markdown("### Strategy 3 trade log")
st.dataframe(trades3.tail(50) if not trades3.empty else pd.DataFrame(), use_container_width=True)