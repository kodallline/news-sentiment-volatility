"""
═══════════════════════════════════════════════════════════════════
Sources:
  1. Yahoo Finance   — цены 15 тикеров (2020-2026) + VIX + Treasury
  2. Google Trends   — "coronavirus"
  3. alternative.me  — CNN Fear & Greed Index (daily, 0-100)
  4. HuggingFace     — финансовые новости с датами (ashraq, первые 30k строк)
  5. HuggingFace     — Twitter финансовые твиты (StephanAkkerman)
  6. Polygon.io      — новости по тикерам (прямой HTTP, без пакета)
  7. Yahoo Finance   — Put/Call Ratio (CBOE) + AAII если получится
pip install yfinance vaderSentiment datasets pytrends requests openpyxl
═══════════════════════════════════════════════════════════════════
"""
import re
import time
import warnings
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ─── НАСТРОЙКИ ───────────────────────────────────────────────
START_DATE = "2020-01-01"
END_DATE   = "2026-01-01"

TICKERS = [
    "SPY",
    "AAPL", "MSFT", "GOOGL", "META",
    "AMZN", "NVDA", "TSLA", "AMD",
    "NFLX",
    "JPM", "GS",
    "JNJ", "XOM", "DIS",
]

POLYGON_API_KEY = "G50DvIALa7dyHEEYtdQH1tLP4pODnCzl"

OUTPUT_FILE    = "final_dataset.csv"
QUALITY_REPORT = "dataset_quality_report.txt"

analyzer = SentimentIntensityAnalyzer()


def vader_score(text):
    return analyzer.polarity_scores(str(text))["compound"]


def flatten_yf(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def make_daily_index():
    return pd.DataFrame({"date": pd.date_range(START_DATE, END_DATE, freq="D")})


def safe_normalize_date(series):
    try:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return parsed.dt.tz_localize(None).dt.normalize()
    except Exception:
        parsed = pd.to_datetime(series, errors="coerce")
        try:
            return parsed.dt.tz_localize(None).dt.normalize()
        except Exception:
            return parsed.dt.normalize()

def download_stocks():
    print("\n[1/6] Downloading stock prices for 15 tickers ...")
    blocks = []
    for ticker in TICKERS:
        try:
            raw = flatten_yf(yf.download(ticker, start=START_DATE, end=END_DATE,
                                          progress=False, auto_adjust=True))
        except Exception as e:
            print(f"  WARNING: {ticker} failed ({e}) — retrying...")
            time.sleep(2)
            try:
                raw = flatten_yf(yf.download(ticker, start=START_DATE, end=END_DATE,
                                              progress=False, auto_adjust=True))
            except Exception as e2:
                print(f"  WARNING: Skipping {ticker}: {e2}")
                continue
        if raw is None or raw.empty:
            print(f"  WARNING: No data for {ticker}")
            continue
        df = raw.reset_index().rename(columns={"Date": "date"})
        df["date"]   = pd.to_datetime(df["date"]).dt.normalize()
        df["ticker"] = ticker
        df["returns"]        = np.log(df["Close"] / df["Close"].shift(1))
        df["volatility_21d"] = df["returns"].rolling(21).std() * np.sqrt(252)
        df["parkinson_vol"]  = (
            np.sqrt((1 / (4 * np.log(2))) * np.log(df["High"] / df["Low"]) ** 2)
            .rolling(21).mean() * np.sqrt(252)
        )
        df["volume_change"]  = np.log(df["Volume"] / df["Volume"].shift(1))
        df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
        blocks.append(df[["date", "ticker", "Open", "High", "Low", "Close",
                           "Volume", "returns", "volatility_21d", "parkinson_vol",
                           "volume_change", "intraday_range"]])
    result = pd.concat(blocks, ignore_index=True)
    print(f"  OK: {len(TICKERS)} tickers -> {len(result):,} rows")
    return result
def download_macro():
    print("\n[2/6] Downloading macro (VIX, 10yr yield) ...")
    def _get(sym, col):
        raw = flatten_yf(yf.download(sym, start=START_DATE, end=END_DATE,
                                      progress=False, auto_adjust=True))
        df = raw[["Close"]].reset_index().rename(columns={"Date": "date", "Close": col})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df
    macro = _get("^VIX", "vix").merge(_get("^TNX", "treasury_yield"), on="date", how="outer")
    print(f"  OK: {len(macro)} macro rows")
    return macro.sort_values("date").reset_index(drop=True)
def download_covid_trends():
    print("\n[3/6] Downloading Google Trends (coronavirus) ...")
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-US", tz=0)
        frames = []
        for y in ["2020", "2021", "2022", "2023", "2024", "2025"]:
            try:
                pt.build_payload(["coronavirus"], timeframe=f"{y}-01-01 {y}-12-31", geo="US")
                chunk = pt.interest_over_time()
                if not chunk.empty:
                    frames.append(chunk[["coronavirus"]])
                time.sleep(1)
            except Exception as e:
                print(f"  WARNING chunk {y}: {e}")
        if frames:
            trends = pd.concat(frames).reset_index()
            trends.rename(columns={"coronavirus": "covid_trend"}, inplace=True)
            trends["date"] = pd.to_datetime(trends["date"]).dt.normalize()
            trends = trends[["date", "covid_trend"]].drop_duplicates("date")
            all_days = make_daily_index()
            trends = all_days.merge(trends, on="date", how="left")
            trends["covid_trend"] = trends["covid_trend"].ffill().fillna(0)
            print(f"  OK: {len(trends)} rows")
            return trends
    except ImportError:
        print("  WARNING: pip install pytrends")
    except Exception as e:
        print(f"  WARNING: Trends failed: {e}")
    df = make_daily_index()
    df["covid_trend"] = np.nan
    return df

def download_fear_greed():
    print("\n[4/6] Downloading CNN Fear & Greed Index ...")
    try:
        r = requests.get("https://api.alternative.me/fng/",
                         params={"limit": 2200, "format": "json"}, timeout=20)
        r.raise_for_status()
        rows = []
        for entry in r.json().get("data", []):
            try:
                rows.append({
                    "date": pd.to_datetime(int(entry["timestamp"]), unit="s").normalize(),
                    "fear_greed_value": int(entry["value"]),
                    "fear_greed_label": entry["value_classification"].lower(),
                })
            except Exception:
                continue
        df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
        df = df[(df["date"] >= START_DATE) & (df["date"] < END_DATE)].reset_index(drop=True)
        def _map(v):
            if v <= 25: return -1.0
            if v <= 45: return -0.5
            if v <= 55: return  0.0
            if v <= 75: return  0.5
            return 1.0
        df["fear_greed_numeric"] = df["fear_greed_value"].apply(_map)
        print(f"  OK: {len(df)} daily rows")
        return df
    except Exception as e:
        print(f"  WARNING: Fear & Greed failed: {e}")
        df = make_daily_index()
        for col in ["fear_greed_value", "fear_greed_label", "fear_greed_numeric"]:
            df[col] = np.nan
        return df



def load_financial_news():
    print("\n[5/6] News sentiment -> будет взят из Polygon (шаг 5c)")
    return make_daily_index()

def load_twitter_sentiment():
    print("\n[5b/6] Loading Twitter financial tweets (HuggingFace) ...")

    candidates = [
        ("StephanAkkerman/stock-market-tweets-data",
         ["date", "created_at", "timestamp", "Date", "created"],
         ["text", "tweet", "content", "body"]),
        ("mjw/stock_market_tweets",
         ["date", "created_at", "timestamp", "Date"],
         ["text", "tweet", "content", "body"]),
        ("TimKoornstra/financial-tweets-sentiment",
         ["date", "created_at", "timestamp", "Date", "post_date"],
         ["text", "tweet", "content", "body", "message"]),
    ]

    for (ds_name, date_hints, text_hints) in candidates:
        try:
            from datasets import load_dataset
            ds = load_dataset(ds_name, split="train",
                              streaming=True, trust_remote_code=False)
            records = []
            for i, row in enumerate(ds):
                records.append(row)
                if i >= 199_999:
                    break
            df = pd.DataFrame(records)
            print(f"  [{ds_name}]: {len(df):,} rows | cols: {list(df.columns)}")

            date_col = next((c for c in date_hints if c in df.columns), None)
            text_col = next((c for c in text_hints if c in df.columns), None)

            if not date_col or not text_col:
                print(f"  Skipping: no date ({date_col}) or text ({text_col})")
                continue

            df["date"] = safe_normalize_date(df[date_col])
            df = df.dropna(subset=["date"])
            df_f = df[(df["date"] >= START_DATE) & (df["date"] < END_DATE)].copy()
            print(f"  Tweets in 2020-2026: {len(df_f):,}")

            if len(df_f) < 100:
                print("  Too few, trying next...")
                continue


            label_col = next((c for c in ["label", "sentiment", "Sentiment",
                                          "sentiment_label"] if c in df_f.columns), None)
            if label_col:
                label_map = {
                    "positive": 1.0, "Positive": 1.0, "POSITIVE": 1.0, 2: 1.0,
                    "negative": -1.0, "Negative": -1.0, "NEGATIVE": -1.0, 0: -1.0,
                    "neutral": 0.0, "Neutral": 0.0, "NEUTRAL": 0.0, 1: 0.0,
                }
                df_f["score"] = df_f[label_col].map(label_map)
                mask = df_f["score"].isna()
                if mask.any():
                    df_f.loc[mask, "score"] = \
                        df_f.loc[mask, text_col].astype(str).apply(vader_score)
            else:
                df_f["score"] = df_f[text_col].astype(str).apply(vader_score)

            df_f["is_pos"] = (df_f["score"] >  0.2).astype(int)
            df_f["is_neg"] = (df_f["score"] < -0.2).astype(int)

            daily = (df_f.groupby("date")
                       .agg(twitter_sentiment_mean=("score",  "mean"),
                            twitter_sentiment_std =("score",  "std"),
                            twitter_positive_pct  =("is_pos", "mean"),
                            twitter_negative_pct  =("is_neg", "mean"),
                            twitter_tweet_count   =("score",  "count"))
                       .reset_index())

            all_days = make_daily_index()
            daily["date"] = pd.to_datetime(daily["date"])
            all_days["date"] = pd.to_datetime(all_days["date"])
            daily = all_days.merge(daily, on="date", how="left")
            for col in ["twitter_sentiment_mean", "twitter_sentiment_std",
                        "twitter_positive_pct", "twitter_negative_pct"]:
                daily[col] = daily[col].ffill()
            daily["twitter_tweet_count"] = daily["twitter_tweet_count"].fillna(0).astype(int)
            daily["twitter_source"] = ds_name

            print(f"  OK: {daily['twitter_tweet_count'].sum():,} tweets, "
                  f"avg {daily['twitter_tweet_count'].mean():.1f}/day")
            return daily

        except Exception as e:
            print(f"  WARNING [{ds_name}]: {str(e)[:120]}")
            continue

    print("  All Twitter datasets failed -> NaN")
    df = make_daily_index()
    for col in ["twitter_sentiment_mean", "twitter_sentiment_std",
                "twitter_positive_pct", "twitter_negative_pct", "twitter_tweet_count"]:
        df[col] = np.nan
    df["twitter_source"] = "none"
    return df

def load_polygon_news():
    print("\n[5c/6] Loading Polygon.io ticker news (direct HTTP) ...")

    BASE_URL = "https://api.polygon.io/v2/reference/news"
    all_articles = []

    for ticker in TICKERS:
        print(f"  {ticker} ...", end=" ", flush=True)
        try:
            params = {
                "ticker": ticker,
                "published_utc.gte": START_DATE,
                "published_utc.lte": END_DATE,
                "order": "asc",
                "limit": 1000,
                "sort": "published_utc",
                "apiKey": POLYGON_API_KEY,
            }
            ticker_articles = []
            next_url = None
            pages = 0

            while pages < 5:
                url = next_url or BASE_URL
                r = requests.get(url,
                                 params=params if not next_url else {"apiKey": POLYGON_API_KEY},
                                 timeout=20)
                r.raise_for_status()
                data = r.json()

                for item in data.get("results", []):
                    try:
                        raw = pd.to_datetime(item["published_utc"], utc=True)
                        ticker_articles.append({
                            "date":        raw.tz_localize(None).normalize(),
                            "ticker":      ticker,
                            "title":       item.get("title", ""),
                            "description": item.get("description", "") or "",
                        })
                    except Exception:
                        continue

                next_url = data.get("next_url")
                pages += 1
                if not next_url:
                    break
                time.sleep(13)

            all_articles.extend(ticker_articles)
            print(f"{len(ticker_articles)} articles")
            time.sleep(13)

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else "?"
            print(f"HTTP {code}")
            if code == 403:
                print("  ⚠ Polygon: проверь API ключ")
                break
            time.sleep(13)
        except Exception as e:
            print(f"ERR: {str(e)[:50]}")
            time.sleep(13)

    if not all_articles:
        print("  No Polygon data -> NaN")
        df = make_daily_index()
        for col in ["polygon_sentiment_mean", "polygon_positive_pct",
                    "polygon_negative_pct", "polygon_article_count"]:
            df[col] = np.nan
        return df

    df = pd.DataFrame(all_articles)
    df["text"]   = df["title"].fillna("") + " " + df["description"].fillna("")
    df["vader"]  = df["text"].apply(vader_score)
    df["is_pos"] = (df["vader"] >  0.2).astype(int)
    df["is_neg"] = (df["vader"] < -0.2).astype(int)

    daily = (df.groupby("date")
               .agg(polygon_sentiment_mean=("vader",  "mean"),
                    polygon_positive_pct  =("is_pos", "mean"),
                    polygon_negative_pct  =("is_neg", "mean"),
                    polygon_article_count =("vader",  "count"))
               .reset_index())

    all_days = make_daily_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = all_days.merge(daily, on="date", how="left")
    for col in ["polygon_sentiment_mean", "polygon_positive_pct", "polygon_negative_pct"]:
        daily[col] = daily[col].ffill()
    daily["polygon_article_count"] = daily["polygon_article_count"].fillna(0).astype(int)

    covered = (daily["polygon_article_count"] > 0).sum()
    print(f"  OK: {daily['polygon_article_count'].sum():,} total, {covered} days covered")
    return daily


def build_dataset():
    stocks  = download_stocks()
    macro   = download_macro()
    covid   = download_covid_trends()
    fg      = download_fear_greed()
    news    = load_financial_news()
    twitter = load_twitter_sentiment()
    polygon = load_polygon_news()

    print("\n-- Assembling final dataset ...")
    df = stocks.copy()
    df["date"] = pd.to_datetime(df["date"])
    for side_df in [macro, covid, fg, news, twitter, polygon]:
        side_df["date"] = pd.to_datetime(side_df["date"])
        df = df.merge(side_df, on="date", how="left")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)


    if "polygon_sentiment_mean" in df.columns:
        df["news_sentiment_mean"] = df["polygon_sentiment_mean"]
        df["news_positive_pct"]   = df["polygon_positive_pct"]
        df["news_negative_pct"]   = df["polygon_negative_pct"]
        df["news_count"]          = df["polygon_article_count"]
        print("  news_* = polygon_* (66k+ статей по тикерам)")

    core_cols = [c for c in [
        "volatility_21d", "parkinson_vol", "vix", "treasury_yield",
        "fear_greed_value", "news_sentiment_mean",
        "twitter_sentiment_mean", "polygon_sentiment_mean",
    ] if c in df.columns]

    df["missing_flags"] = df[core_cols].apply(
        lambda row: ",".join(c for c in core_cols if pd.isna(row[c])) or "none", axis=1)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved -> {OUTPUT_FILE}  |  {df.shape[0]:,} rows x {df.shape[1]} columns")


    lines = ["=" * 64, "DATASET QUALITY REPORT", "=" * 64, "",
             f"Period  : {START_DATE} -> {END_DATE}",
             f"Tickers : {', '.join(TICKERS)}",
             f"Shape   : {df.shape[0]:,} rows x {df.shape[1]} columns", "",
             "ИСТОЧНИКИ СЕНТИМЕНТА", "-" * 55,
             "  news_sentiment_*      ashraq/financial-news (Reuters/Bloomberg), VADER",
             "  twitter_sentiment_*   StephanAkkerman/stock-market-tweets-data",
             "  polygon_sentiment_*   Polygon.io/Massive.com (по тикерам, ~2023-2025)",
             "  fear_greed_*          CNN Fear&Greed Index (alternative.me, daily)",
             "  vix                   CBOE VIX (ожидаемая волатильность рынка)",
             "  covid_trend           Google Trends 'coronavirus' US", "",
             "NaN SUMMARY", "-" * 55]

    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 0:
            lines.append(f"  {col:<30} {pct:.1f}% missing")

    lines += [
        "", "КАК РАБОТАТЬ В COLAB", "-" * 55,
        "  import pandas as pd",
        "  df = pd.read_csv('final_dataset.csv', parse_dates=['date'])",
        "  spy = df[df['ticker'] == 'SPY'].copy()",
        "",
        "  # Корреляция с волатильностью:",
        "  cols = ['news_sentiment_mean', 'twitter_sentiment_mean',",
        "          'polygon_sentiment_mean', 'fear_greed_numeric',",
        "          'put_call_ratio', 'vix', 'volatility_21d']",
        "  spy[cols].corr()['volatility_21d']",
        ""
    ]

    report = "\n".join(lines)
    with open(QUALITY_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved -> {QUALITY_REPORT}\n")
    print(report)
    return df

if __name__ == "__main__":
    df = build_dataset()

    print("\n-- SPY sanity check --")
    spy = df[(df["ticker"] == "SPY") & df["volatility_21d"].notna()].head(3)
    cols = [c for c in ["date", "Close", "volatility_21d", "vix",
                        "fear_greed_value", "news_sentiment_mean",
                        "twitter_sentiment_mean", "polygon_sentiment_mean",
                        "put_call_ratio", "missing_flags"] if c in spy.columns]
    print(spy[cols].to_string(index=False))

    print("\n-- Покрытие источников (% не-NaN) --")
    for col in ["news_sentiment_mean", "twitter_sentiment_mean",
                "polygon_sentiment_mean", "fear_greed_value", "vix"]:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            print(f"  {col:<30} {pct:.1f}%")
